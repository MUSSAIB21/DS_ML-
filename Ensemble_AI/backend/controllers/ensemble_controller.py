from sentence_transformers import SentenceTransformer
import httpx
import numpy as np
import re
import asyncio
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# Load embedding model once at startup
# Using paraphrase model which is better at detecting semantic equivalence
# between different phrasings of the same answer (e.g., "Paris" vs "The capital is Paris")
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Model configuration
MODEL_NAMES = ["llama3", "mistral", "phi3"]
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 60.0


# --------------------------------------------------------
# ASYNC OLLAMA CALLS (PARALLEL EXECUTION)
# --------------------------------------------------------

async def ask_ollama(model_name: str, prompt: str) -> Dict[str, str]:
    """
    Query a single Ollama model asynchronously with chain-of-thought prompting.

    Returns dict with 'answer' and 'reasoning' fields.
    """
    # Chain-of-thought prompt that asks model to explain its certainty
    full_prompt = f"""You are a concise expert assistant.
Answer the question in one sentence, then briefly explain your level of certainty.

Format your response exactly as:
ANSWER: [your one-sentence answer]
REASONING: [one sentence explaining if you're certain, uncertain, or why]

Question: {prompt}"""

    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False,
        "temperature": 0.0,  # Deterministic for consistency
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(OLLAMA_BASE_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            full_response = str(data.get("response", "")).strip()

            # Parse the structured response
            answer, reasoning = parse_cot_response(full_response)

            return {
                "answer": answer,
                "reasoning": reasoning,
                "model": model_name
            }

    except httpx.TimeoutException:
        logger.error(f"Timeout querying {model_name}")
        return {
            "answer": f"[Timeout from {model_name}]",
            "reasoning": "Request timed out",
            "model": model_name,
            "error": True
        }
    except Exception as e:
        logger.error(f"Error querying {model_name}: {str(e)}")
        return {
            "answer": f"[Error from {model_name}]",
            "reasoning": str(e),
            "model": model_name,
            "error": True
        }


def parse_cot_response(response: str) -> Tuple[str, str]:
    """
    Extract ANSWER and REASONING from chain-of-thought response.

    Handles cases where model doesn't follow format perfectly.
    """
    answer = ""
    reasoning = ""

    # Try to extract structured format
    answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)

    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Fallback: use first sentence as answer
        sentences = re.split(r'[.!?]\s+', response)
        answer = sentences[0] if sentences else response

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        # Fallback: use rest of response as reasoning
        reasoning = response.replace(answer, "").strip()

    return answer, reasoning


async def query_all_models(question: str) -> List[Dict[str, str]]:
    """
    Query all models in parallel for speed.

    Returns list of response dicts with answer and reasoning.
    """
    # Create async tasks for all models
    tasks = [ask_ollama(model, question) for model in MODEL_NAMES]

    # Run them concurrently
    results = await asyncio.gather(*tasks)

    # Filter out error responses
    valid_results = [r for r in results if not r.get("error", False)]

    if len(valid_results) == 0:
        raise RuntimeError("All models failed to respond")

    if len(valid_results) < len(MODEL_NAMES):
        logger.warning(f"Only {len(valid_results)}/{len(MODEL_NAMES)} models responded")

    return valid_results


# --------------------------------------------------------
# TEXT NORMALIZATION
# --------------------------------------------------------

def normalize_answer(text: str) -> str:
    """
    Clean and standardize text for embedding comparison.

    Removes filler phrases, extracts key entities, and normalizes answer formats
    so that "Paris" and "The capital of France is Paris" are treated as equivalent.
    """
    if not isinstance(text, str):
        text = str(text)

    if not text:
        return ""

    text = text.lower()

    # Remove common AI filler phrases that don't add semantic value
    filler_patterns = [
        r"as an ai language model[, ]*",
        r"as a language model[, ]*",
        r"as an ai[, ]*",
        r"in summary[, ]*",
        r"to summarize[, ]*",
        r"it's important to note that[, ]*",
        r"overall[, ]*",
        r"in conclusion[, ]*",
    ]

    for pattern in filler_patterns:
        text = re.sub(pattern, "", text)

    # Extract key answer from common question-answer templates
    # This helps "The capital of France is Paris" match with just "Paris"
    qa_templates = [
        r"the (capital|answer|result|solution) (of|to|for) .+ is ",
        r"the (capital|answer|result|solution) is ",
        r".+ (is|are|was|were) (called|named|known as) ",
        r"it (is|was) ",
        r"this (is|was) ",
        r"that (is|was) ",
    ]

    for template in qa_templates:
        text = re.sub(template, "", text, flags=re.IGNORECASE)

    # Remove leading articles for better matching
    text = re.sub(r"^(the|a|an) ", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove trailing punctuation
    text = re.sub(r"[.,;!?]+$", "", text)

    # For very short answers (1-3 words), return as-is
    # These are likely the core answer already
    word_count = len(text.split())
    if word_count <= 3:
        return text

    # For longer answers, limit to first 3 sentences to focus on core answer
    sentences = re.split(r'(?<=[.!?]) +', text)
    text = " ".join(sentences[:3])

    # Limit total words to prevent embedding bias from length
    words = text.split()
    if len(words) > 80:
        text = " ".join(words[:80])

    return text


# --------------------------------------------------------
# EMBEDDING AND SIMILARITY
# --------------------------------------------------------

def embed(text: str) -> np.ndarray:
    """
    Convert text to semantic embedding vector.

    Uses normalized text to ensure consistent comparisons.
    """
    cleaned = normalize_answer(text)
    return embedding_model.encode(cleaned)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embedding vectors.

    Returns value between 0 (completely different) and 1 (identical).
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# --------------------------------------------------------
# CONSENSUS SCORE CALCULATION
# --------------------------------------------------------

def compute_consensus_score(answers: List[str]) -> Tuple[str, float, np.ndarray, List[np.ndarray]]:
    """
    Find the answer with highest agreement among models.

    Returns:
        - Best answer (the one most similar to others)
        - Consensus score (0-100)
        - Similarity matrix
        - Embeddings for all answers
    """
    # Convert all answers to embeddings
    embeddings = [embed(ans) for ans in answers]
    n = len(embeddings)

    # Build pairwise similarity matrix
    sim_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sim_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

    # Find answer most similar to all others (highest average similarity)
    avg_similarities = sim_matrix.mean(axis=1)
    best_index = int(np.argmax(avg_similarities))
    final_answer = answers[best_index]

    # Calculate consensus score based on agreement
    # Count how many models gave similar answers (similarity >= 0.7)
    threshold = 0.7
    agreements = sum(
        1 for i in range(n)
        if sim_matrix[best_index][i] >= threshold
    )

    agreement_ratio = agreements / n
    similarity_strength = float(avg_similarities[best_index])

    # Consensus score: blend of agreement ratio and similarity strength
    consensus_score = (agreement_ratio * similarity_strength) * 100.0

    return final_answer, consensus_score, sim_matrix, embeddings


# --------------------------------------------------------
# CONFIDENCE SCORE FROM REASONING
# --------------------------------------------------------

def extract_confidence_from_reasoning(reasoning: str) -> float:
    """
    Analyze model's reasoning to detect uncertainty.

    Returns confidence penalty between 0.0 (very uncertain) and 1.0 (very certain).
    Models often reveal their own uncertainty through hedging language.
    """
    if not reasoning:
        return 0.8  # Default moderate confidence if no reasoning provided

    reasoning_lower = reasoning.lower()

    # Strong uncertainty signals - model explicitly admits it doesn't know
    strong_uncertainty = [
        "unsure", "uncertain", "not sure", "don't know", "do not know",
        "unclear", "cannot say", "can't say", "not certain",
        "might be wrong", "could be incorrect", "possibly incorrect"
    ]

    # Medium uncertainty signals - model is hedging
    medium_uncertainty = [
        "possibly", "perhaps", "might", "may", "could be",
        "i think", "i believe", "i suspect", "probably", "likely",
        "seems like", "appears to be", "tends to"
    ]

    # Qualifier words that indicate nuance or debate
    qualifiers = [
        "though", "however", "but", "although", "while",
        "debated", "controversial", "disputed", "unclear",
        "commonly believed", "often thought"
    ]

    # Certainty indicators - model is confident
    certainty_indicators = [
        "definitely", "certainly", "absolutely", "clearly",
        "without doubt", "for sure", "undoubtedly", "obviously",
        "well-established", "proven", "confirmed", "verified"
    ]

    # Check for uncertainty markers
    if any(marker in reasoning_lower for marker in strong_uncertainty):
        return 0.3  # Very low confidence

    if any(marker in reasoning_lower for marker in medium_uncertainty):
        # Count how many hedges appear
        hedge_count = sum(1 for marker in medium_uncertainty if marker in reasoning_lower)
        return max(0.4, 0.7 - (hedge_count * 0.1))  # Reduce for multiple hedges

    if any(qual in reasoning_lower for qual in qualifiers):
        return 0.75  # Slight reduction for qualifiers

    if any(indicator in reasoning_lower for indicator in certainty_indicators):
        return 1.0  # Full confidence

    # No strong signals either way
    return 0.85  # Default good confidence


def compute_confidence_score(responses: List[Dict[str, str]]) -> Tuple[float, List[float]]:
    """
    Calculate overall confidence from all model reasoning.

    Returns:
        - Average confidence score (0-100)
        - Individual confidence scores for each model
    """
    individual_confidences = []

    for response in responses:
        reasoning = response.get("reasoning", "")
        confidence_penalty = extract_confidence_from_reasoning(reasoning)
        individual_confidences.append(confidence_penalty)

    # Average confidence across all models
    avg_confidence = np.mean(individual_confidences)
    confidence_score = avg_confidence * 100.0

    return confidence_score, individual_confidences


# --------------------------------------------------------
# RELIABILITY ASSESSMENT
# --------------------------------------------------------

def assess_reliability(consensus_score: float, confidence_score: float) -> Tuple[str, int, str]:
    """
    Combine consensus and confidence into overall reliability rating.

    Returns:
        - Reliability level (VERY_RELIABLE, RELIABLE, etc.)
        - Star rating (1-5)
        - Warning message (if applicable)
    """
    warning = None

    # High consensus + High confidence = Very reliable
    if consensus_score >= 80 and confidence_score >= 80:
        return "VERY_RELIABLE", 5, None

    # High consensus + Medium confidence = Reliable
    if consensus_score >= 80 and confidence_score >= 50:
        return "RELIABLE", 4, None

    # High consensus + Low confidence = Questionable (red flag!)
    if consensus_score >= 80 and confidence_score < 50:
        warning = "Models agree but express uncertainty in their reasoning"
        return "QUESTIONABLE", 2, warning

    # Medium consensus + High confidence = Moderate (possibly subjective question)
    if consensus_score >= 50 and confidence_score >= 80:
        warning = "Models disagree despite being confident - may be subjective"
        return "MODERATE", 3, warning

    # Medium consensus + Medium confidence = Moderate
    if consensus_score >= 50:
        return "MODERATE", 3, None

    # Low consensus = Unreliable regardless of confidence
    warning = "Models fundamentally disagree on the answer"
    return "UNRELIABLE", 1, warning


# --------------------------------------------------------
# PUBLIC ENTRY POINT
# --------------------------------------------------------

async def ask_ensemble(question: str) -> Dict:
    """
    Main ensemble query function with dual metrics.

    Process:
    1. Query all models in parallel with chain-of-thought prompts
    2. Calculate consensus score (how much models agree)
    3. Calculate confidence score (how certain models are)
    4. Assess overall reliability

    Returns comprehensive result with both metrics and per-model details.
    """
    # Query all models concurrently
    responses = await query_all_models(question)

    # Extract just the answers for consensus calculation
    answers = [r["answer"] for r in responses]

    # Calculate consensus score (semantic agreement between answers)
    final_answer, consensus_score, sim_matrix, embeddings = compute_consensus_score(answers)

    # Calculate confidence score (uncertainty from reasoning)
    confidence_score, individual_confidences = compute_confidence_score(responses)

    # Assess overall reliability
    reliability, stars, warning = assess_reliability(consensus_score, confidence_score)

    # Build per-model details for transparency
    final_embedding = embed(final_answer)
    model_details = []

    for i, response in enumerate(responses):
        # How similar is this model's answer to the final consensus answer
        consensus_contribution = cosine_similarity(final_embedding, embeddings[i]) * 100

        model_details.append({
            "name": response["model"],
            "answer": response["answer"],
            "reasoning": response["reasoning"],
            "consensus_contribution": round(consensus_contribution, 2),
            "confidence_level": round(individual_confidences[i] * 100, 2)
        })

    # Construct response
    result = {
        "question": question,
        "final_answer": final_answer,

        # Dual metrics
        "consensus_score": round(consensus_score, 2),
        "consensus_explanation": "Measures how much models agree (semantic similarity)",

        "confidence_score": round(confidence_score, 2),
        "confidence_explanation": "Measures how certain models are (from reasoning)",

        # Combined assessment
        "reliability": reliability,
        "reliability_stars": stars,
        "warning": warning,

        # Detailed breakdown
        "models": model_details,

        # Raw data for debugging
        "similarity_matrix": sim_matrix.tolist(),
    }

    return result
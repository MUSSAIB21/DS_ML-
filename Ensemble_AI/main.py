from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from backend.controllers.ensemble_controller import ask_ensemble
import logging
import time
from typing import Optional

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Ensemble API",
    description="Query multiple AI models and get consensus answers with confidence scores",
    version="2.0.0"
)


class AskPayload(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The question to ask the ensemble"
    )
    use_cache: Optional[bool] = Field(
        default=True,
        description="Whether to use cached results for identical questions"
    )

    @validator('question')
    def sanitize_question(cls, v):
        # Remove potential prompt injection attempts
        dangerous_patterns = [
            'ignore previous',
            'ignore all previous',
            'disregard',
            'new instructions',
            'system:',
            '<|im_start|>',
            '<|im_end|>',
        ]

        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in v_lower:
                raise ValueError(f"Question contains potentially unsafe pattern: {pattern}")

        # Basic cleanup
        v = v.strip()

        return v


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log request details for monitoring
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        f"path={request.url.path} method={request.method} "
        f"status={response.status_code} duration={duration:.3f}s"
    )

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Catch all exceptions and return structured error responses
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "detail": str(exc) if app.debug else None
        }
    )


@app.get("/")
async def root():
    return {
        "service": "LLM Ensemble API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "ask": "/ask",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    # Basic health check endpoint
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


@app.post("/ask")
async def ask(payload: AskPayload):
    """
    Query the ensemble with a question.

    Returns consensus answer with dual metrics:
    - consensus_score: How much models agree (semantic similarity)
    - confidence_score: How certain models are (from reasoning)
    - reliability: Combined assessment of answer quality
    """
    try:
        logger.info(f"Processing question: {payload.question[:100]}")

        result = await ask_ensemble(payload.question)

        logger.info(
            f"Question answered - consensus={result['consensus_score']} "
            f"confidence={result['confidence_score']} "
            f"reliability={result['reliability']}"
        )

        return result

    except ValueError as e:
        # Input validation errors
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected errors
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process question. Please try again."
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
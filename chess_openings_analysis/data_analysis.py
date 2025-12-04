# This file contains functions for processing chess data.
import pandas as pd


def load_and_clean_data(filepath):
    """
    Loads chess data and cleans it up.
    Handles missing values and weird data points.

    """
    df = pd.read_csv(filepath)

    # Print initial data info (like in a notebook)
    print(f"Initial shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Detect which format we have and rename columns accordingly
    column_mapping = {}

    # Lichess format
    if 'WhiteElo' in df.columns:
        column_mapping = {
            'WhiteElo': 'white_elo',
            'BlackElo': 'black_elo',
            'ECO': 'opening_eco',
            'Opening': 'opening_name',
            'Result': 'result',
            'TotalMoves': 'move_count',
            'Moves': 'moves'
        }
    # Chess.com format
    elif 'white_rating' in df.columns:
        column_mapping = {
            'white_rating': 'white_elo',
            'black_rating': 'black_elo',
            'Result': 'result',
            'ECO': 'opening_eco',
            'Opening': 'opening_name',
            'rated': 'is_rated',
            'turns': 'move_count'
        }
    # Already standardized format
    elif 'white_elo' in df.columns:
        column_mapping = {
            'Result': 'result',
            'ECO': 'opening_eco',
            'Opening': 'opening_name',
            'turns': 'move_count'
        }

    df = df.rename(columns=column_mapping)

    # Create 'result' column from 'winner' if it doesn't exist
    if 'result' not in df.columns and 'winner' in df.columns:
        result_map = {
            'white': '1-0',
            'black': '0-1',
            'draw': '1/2-1/2'
        }
        df['result'] = df['winner'].map(result_map)
        print("Created 'result' column from 'winner' column")

    # Create move_count from turns if needed
    if 'move_count' not in df.columns and 'turns' in df.columns:
        df['move_count'] = df['turns']

    # Check what columns we have now
    print(
        f"\nAfter renaming, columns include: {[col for col in ['white_elo', 'black_elo', 'result', 'opening_eco'] if col in df.columns]}")

    # Drop rows with missing critical data
    initial_len = len(df)

    # Check which columns exist before dropping
    required_cols = ['white_elo', 'black_elo', 'opening_eco', 'result']
    existing_cols = [col for col in required_cols if col in df.columns]

    if existing_cols:
        df = df.dropna(subset=existing_cols)

    # Convert ELO to numeric
    if 'white_elo' in df.columns:
        df['white_elo'] = pd.to_numeric(df['white_elo'], errors='coerce')
    if 'black_elo' in df.columns:
        df['black_elo'] = pd.to_numeric(df['black_elo'], errors='coerce')

    # Removing unrealistic ELO ratings
    if 'white_elo' in df.columns and 'black_elo' in df.columns:
        df = df[(df['white_elo'] >= 800) & (df['white_elo'] <= 3000)]
        df = df[(df['black_elo'] >= 800) & (df['black_elo'] <= 3000)]

    # Removing games with too few moves(at least 10 moves = 20 plies)
    if 'move_count' in df.columns:
        df['move_count'] = pd.to_numeric(df['move_count'], errors='coerce')
        df = df[df['move_count'] >= 10]

    print(f"Removed {initial_len - len(df)} rows with bad data")
    print(f"Final shape: {df.shape}")

    return df


def simplified_eco_classifier(eco_code):
    # Simple opening classification based on ECO code.

    eco_str = str(eco_code).upper().strip()

    if eco_str.startswith('A'):
        return "Flank Openings (A00-A39)"
    elif eco_str.startswith('B'):
        return "Sicilian & French (B00-B99)"
    elif eco_str.startswith('C'):
        if eco_str[1:] < '20':
            return "King's Knight Openings (C00-C19)"
        else:
            return "Italian & Scotch (C20-C99)"
    elif eco_str.startswith('D'):
        if eco_str < 'D30':
            return "Queen's Gambit (D00-D29)"
        else:
            return "Closed Games (D30-D99)"
    elif eco_str.startswith('E'):
        return "Indian Defenses (E00-E99)"
    else:
        return "Unknown Opening"


def extract_openings(df):
    # Extracts opening names from ECO codes.

    print("Classifying openings...")

    # Use the ECO code if opening name is missing
    if 'opening_name' not in df.columns or df['opening_name'].isna().all():
        df['opening_name'] = df['opening_eco'].apply(simplified_eco_classifier)
    else:
        # Fill missing opening names with ECO classification
        missing_mask = df['opening_name'].isna()
        df.loc[missing_mask, 'opening_name'] = df.loc[missing_mask, 'opening_eco'].apply(simplified_eco_classifier)

    # Clean up opening names
    df['opening_name'] = df['opening_name'].fillna('Unknown')

    print(f"Top 5 openings:\n{df['opening_name'].value_counts().head()}")

    return df


def create_elo_categories(df):
    """
    Creates ELO rating categories.
    Used for analyzing different skill levels.
    """
    # Calculate average ELO per game
    df['avg_elo'] = (df['white_elo'] + df['black_elo']) / 2

    # Create categories (common thresholds found online)
    bins = [0, 1200, 1400, 1600, 2000, 3000]
    labels = ['Beginner (<1200)', 'Novice (1200-1400)', 'Intermediate (1400-1600)',
              'Advanced (1600-2000)', 'Expert (>2000)']

    df['elo_category'] = pd.cut(df['avg_elo'], bins=bins, labels=labels, right=False)

    print("\nELO distribution:")
    print(df['elo_category'].value_counts())

    return df


def get_opening_recommendation(elo_rating, opening_stats):

    #Simple function to get opening recommendations.

    category = ""
    if elo_rating < 1200:
        category = 'Beginner (<1200)'
    elif elo_rating < 1400:
        category = 'Novice (1200-1400)'
    elif elo_rating < 1600:
        category = 'Intermediate (1400-1600)'
    elif elo_rating < 2000:
        category = 'Advanced (1600-2000)'
    else:
        category = 'Expert (>2000)'

    # Filter for the player's category
    category_stats = opening_stats[opening_stats['elo_category'] == category]

    if category_stats.empty:
        return "Not enough data for this rating level"

    # Find opening with the highest white win rate
    best_opening = category_stats.loc[category_stats['white_win_rate'].idxmax()]

    return {
        'elo_category': category,
        'recommended_opening': best_opening['opening_name'],
        'white_win_rate': best_opening['white_win_rate'],
        'total_games': best_opening['total_games']
    }

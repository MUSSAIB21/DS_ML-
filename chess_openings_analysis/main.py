import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

from data_analysis import load_and_clean_data, extract_openings, create_elo_categories
from visualize import plot_win_rates, plot_feature_importance

# Configuration
DATA_PATH = "data/games_metadata_profile_2024_01.csv"
PROCESSED_PATH = "data/processed_games.csv"
MODEL_PATH = "models/opening_predictor.pkl"
RESULTS_DIR = "results"
N_GAMES = 130000


def create_sample_data():
    # Generates data if none available
    np.random.seed(42)

    data = {
        'white_elo': np.random.randint(800, 2500, N_GAMES),
        'black_elo': np.random.randint(800, 2500, N_GAMES),
        'opening_eco': np.random.choice(['B20', 'C50', 'D00', 'E60', 'C45'], N_GAMES),
        'opening_name': np.random.choice(['Sicilian', 'Italian Game', 'Queen Pawn', 'King Indian', 'Scotch'], N_GAMES),
        'result': np.random.choice(['1-0', '0-1', '1/2-1/2'], N_GAMES, p=[0.4, 0.4, 0.2]),
        'moves': ['1. e4 e5 2. Nf3'] * N_GAMES  # Dummy moves
    }

    df = pd.DataFrame(data)
    df.to_csv(DATA_PATH, index=False)
    print(f"Created sample data with {N_GAMES} games")


def train_model(df):

    #Trains a Random Forest to predict game outcomes

    print("\nTraining ML model...")


    # Features
    df['avg_elo'] = (df['white_elo'] + df['black_elo']) / 2
    df['elo_diff'] = df['white_elo'] - df['black_elo']

    #Encode opening as numeric
    opening_codes = pd.Categorical(df['opening_eco']).codes
    df['opening_code'] = opening_codes

    #Define features and target
    features = ['avg_elo', 'elo_diff', 'opening_code']
    x = df[features]

    #Convert results to binary: 1 if White wins, 0 otherwise
    y = (df['result'] == '1-0').astype(int)

    #Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Black Wins/Draw', 'White Wins']))

    # Save model
    os.makedirs('models', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    return model, features


def analyze_openings_by_elo(df):
    """
    Core analysis: calculates win rates for each opening at different ELO levels.
    This is the main research going on here
    """
    print("\nAnalyzing openings by ELO rating...")

    # Create ELO categories
    df = create_elo_categories(df)

    # Calculate win rates for each opening and ELO category
    opening_stats = df.groupby(['elo_category', 'opening_name']).agg({
        'result': [
            lambda x: (x == '1-0').sum(),  # White wins
            lambda x: (x == '0-1').sum(),  # Black wins
            lambda x: (x == '1/2-1/2').sum(),  # Draws
            'count'
        ]
    }).round(2)

    # Flatten column names
    opening_stats.columns = ['white_wins', 'black_wins', 'draws', 'total_games']
    opening_stats = opening_stats.reset_index()

    # Calculate win rates
    opening_stats['white_win_rate'] = opening_stats['white_wins'] / opening_stats['total_games']
    opening_stats['black_win_rate'] = opening_stats['black_wins'] / opening_stats['total_games']
    opening_stats['draw_rate'] = opening_stats['draws'] / opening_stats['total_games']

    # Filter out openings with too few games
    opening_stats = opening_stats[opening_stats['total_games'] >= 100]

    return opening_stats


def main():
    print("=" * 60)
    print("CHESS OPENINGS ANALYSIS - FINAL PROJECT")
    print(f"Analyzing {N_GAMES:,} games")
    print("=" * 60)

    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Step 1: Get data
    if not os.path.exists(DATA_PATH):
        print("No data found at", DATA_PATH)

        # Check for data file
        if os.path.exists('data/games_metadata_profile_2024_01.csv'):
            print("Found games_metadata_profile_2024_01.csv, converting to standard format...")
            print(f"Loading {N_GAMES:,} games from dataset...")

            df = pd.read_csv('data/games_metadata_profile_2024_01.csv', nrows=N_GAMES)

            # Rename columns to match expected format
            column_mapping = {
                'white_rating': 'white_elo',
                'black_rating': 'black_elo',
                'WhiteElo': 'white_elo',
                'BlackElo': 'black_elo',
                'Result': 'result',
                'ECO': 'opening_eco',
                'Opening': 'opening_name',
                'rated': 'is_rated',
                'turns': 'move_count'
            }
            df = df.rename(columns=column_mapping)

            # Create result column from winner if needed
            if 'result' not in df.columns and 'winner' in df.columns:
                result_map = {
                    'white': '1-0',
                    'black': '0-1',
                    'draw': '1/2-1/2'
                }
                df['result'] = df['winner'].map(result_map)

            df.to_csv(DATA_PATH, index=False)
            print(f"Converted {len(df):,} games to {DATA_PATH}")
        else:
            print("\n" + "="*60)
            print("ERROR: No chess dataset found!")
            print("="*60)
            print("\nPlease download the chess dataset manually:")
            print("1. Go to: https://www.kaggle.com/datasets/shkarupylomaxim/chess-games-dataset-lichess-2017-may")
            print("2. Click 'Download' button")
            print("3. Extract the zip file")
            print("4. Place 'games_metadata_profile_2024_01.csv' in your 'data/' folder")
            print("\nYour data folder should look like:")
            print("  data/")
            print("    └── games_metadata_profile_2024_01.csv")
            print("="*60)
            return

    # Step 2: Load and clean data
    print("\nStep 1: Loading data...")
    df = load_and_clean_data(DATA_PATH)
    print(f"Loaded {len(df):,} games")

    # Step 3: Extract openings
    print("\nStep 2: Extracting openings...")
    df = extract_openings(df)
    print(f"Found {df['opening_name'].nunique():,} unique openings")

    # Step 4: Train ML model
    print("\nStep 3: Training model...")
    model, features = train_model(df)

    # Step 5: Analyze openings by ELO
    print("\nStep 4: Analyzing openings...")
    opening_stats = analyze_openings_by_elo(df)
    print(f"Analyzing {len(opening_stats)} opening-category combinations")
    print(f"(Filtered to openings with ≥100 games for reliability)")

    # Step 6: Create visualizations
    print("\nStep 5: Creating visualizations...")
    plot_win_rates(opening_stats)
    plot_feature_importance(model, features)

    # Step 7: Save results
    opening_stats.to_csv('results/opening_analysis.csv', index=False)
    print(f"\n✓ Analysis complete! Check the 'results/' folder for charts.")
    print(f"✓ Opening statistics saved to results/opening_analysis.csv")
    print(f"✓ Model saved to {MODEL_PATH}")

    # Print top findings
    print("\n" + "=" * 60)
    print("TOP FINDINGS:")
    print("=" * 60)

    # Overall statistics
    total_games = opening_stats['total_games'].sum()
    avg_games = opening_stats['total_games'].mean()
    print(f"\nTotal games analyzed: {total_games:,}")
    print(f"Average games per opening: {avg_games:.0f}")

    # Best opening for beginners
    beginner_data = opening_stats[opening_stats['elo_category'] == 'Beginner (<1200)']
    if not beginner_data.empty:
        best_for_beginners = beginner_data.loc[beginner_data['white_win_rate'].idxmax()]
        print(f"\nBest opening for beginners: {best_for_beginners['opening_name']}")
        print(f"  White win rate: {best_for_beginners['white_win_rate']:.1%}")
        print(f"  Based on {int(best_for_beginners['total_games'])} games")

    # Best opening for intermediate
    intermediate_data = opening_stats[opening_stats['elo_category'] == 'Intermediate (1400-1600)']
    if not intermediate_data.empty:
        best_for_intermediate = intermediate_data.loc[intermediate_data['white_win_rate'].idxmax()]
        print(f"\nBest opening for intermediate: {best_for_intermediate['opening_name']}")
        print(f"  White win rate: {best_for_intermediate['white_win_rate']:.1%}")
        print(f"  Based on {int(best_for_intermediate['total_games'])} games")

    # Best opening for advanced
    advanced_data = opening_stats[opening_stats['elo_category'] == 'Advanced (1600-2000)']
    if not advanced_data.empty:
        best_for_advanced = advanced_data.loc[advanced_data['white_win_rate'].idxmax()]
        print(f"\nBest opening for advanced: {best_for_advanced['opening_name']}")
        print(f"  White win rate: {best_for_advanced['white_win_rate']:.1%}")
        print(f"  Based on {int(best_for_advanced['total_games'])} games")

    # Most reliable findings (most games)
    print(f"\n" + "=" * 60)
    print("MOST RELIABLE OPENINGS (highest sample size):")
    print("=" * 60)
    top_reliable = opening_stats.nlargest(5, 'total_games')
    for idx, row in top_reliable.iterrows():
        print(f"\n{row['opening_name']} ({row['elo_category']})")
        print(f"  White win rate: {row['white_win_rate']:.1%}")
        print(f"  Sample size: {int(row['total_games'])} games")


if __name__ == "__main__":

    main()

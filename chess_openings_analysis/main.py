import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

from data_analysis import load_and_clean_data, extract_openings, create_elo_categories
from visualize import plot_win_rates, plot_feature_importance, plot_opening_popularity, create_summary_report

# Configuration
DATA_PATH = "data/games_metadata_profile_2024_01.csv"
PROCESSED_PATH = "data/processed_games.csv"
MODEL_PATH = "models/opening_predictor.pkl"
RESULTS_DIR = "results"
N_GAMES = 130000



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

        if os.path.exists('data/games_metadata_profile_2024_01.csv'):
            print("Found dataset, converting to standard format...")
            df = pd.read_csv('data/games_metadata_profile_2024_01.csv', nrows=N_GAMES)

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

            if 'result' not in df.columns and 'winner' in df.columns:
                result_map = {'white': '1-0', 'black': '0-1', 'draw': '1/2-1/2'}
                df['result'] = df['winner'].map(result_map)

            df.to_csv(DATA_PATH, index=False)
            print(f"Converted {len(df):,} games to {DATA_PATH}")
        else:
            print("\nERROR: No chess dataset found!")
            print("Please download from Kaggle and place in 'data/' folder.")
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

    # Step 6: Create visualizations
    print("\nStep 5: Creating visualizations...")
    plot_win_rates(opening_stats)
    plot_feature_importance(model, features)
    plot_opening_popularity(df)  # ✅ integrated

    # Step 7: Save results
    opening_stats.to_csv('results/opening_analysis.csv', index=False)
    create_summary_report(opening_stats, df)  # ✅ integrated

    print("\n✓ Analysis complete! Check the 'results/' folder for charts and reports.")

if __name__ == "__main__":

    main()

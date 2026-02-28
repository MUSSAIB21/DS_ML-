import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

from data_analysis import load_and_clean_data, extract_openings, create_elo_categories
from visualize import plot_win_rates, plot_feature_importance, plot_confusion_matrix

# Configuration
DATA_PATH = "data/games_metadata_profile_2024_01.csv"
PROCESSED_PATH = "data/processed_games_metadata_profile_2024_01.csv"
MODEL_PATH = "models/opening_predictor.pkl"
RESULTS_DIR = "results"
N_GAMES = 130000


def train_model(df):
    """
    Trains both Random Forest and Logistic Regression to predict game outcomes.
    """
    print("\nTraining ML models...")

    # Features
    df['avg_elo'] = (df['white_elo'] + df['black_elo']) / 2
    df['elo_diff'] = df['white_elo'] - df['black_elo']

    # Encode opening as numeric
    opening_codes = pd.Categorical(df['opening_eco']).codes
    df['opening_code'] = opening_codes

    # Define features and target
    features = ['avg_elo', 'elo_diff', 'opening_code']
    X = df[features]

    # Convert results to binary: 1 if White wins, 0 otherwise
    y = (df['result'] == '1-0').astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # MODEL 1: RANDOM FOREST

    print("\n--- Random Forest Classifier ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred_rf,
        target_names=['Black Wins/Draw', 'White Wins']
    ))


    plot_confusion_matrix(y_test, y_pred_rf, model_name="Random Forest")

    # MODEL 2: LOGISTIC REGRESSION

    print("\n--- Logistic Regression ---")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)

    # Make predictions
    y_pred_lr = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)

    print(f"Logistic Regression Accuracy: {lr_accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred_lr,
        target_names=['Black Wins/Draw', 'White Wins']
    ))
    plot_confusion_matrix(y_test, y_pred_lr, model_name="Logistic Regression")

    # MODEL COMPARISON

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"Random Forest:        {rf_accuracy:.2%}")
    print(f"Logistic Regression:  {lr_accuracy:.2%}")
    print(f"Difference:           {abs(rf_accuracy - lr_accuracy):.2%}")

    if rf_accuracy > lr_accuracy:
        print("\n✓ Random Forest performs better")
        print("  (Better at capturing non-linear relationships)")
    else:
        print("\n✓ Logistic Regression performs better")
        print("  (More interpretable, captures linear trends)")


    # FEATURE IMPORTANCE (Random Forest only)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("=" * 60)
    importances = rf_model.feature_importances_
    for feature, importance in zip(features, importances):
        print(f"{feature:15s}: {importance:.3f}")

    # Save both models
    os.makedirs('models', exist_ok=True)

    with open('models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

    with open('models/logistic_regression.pkl', 'wb') as f:
        pickle.dump(lr_model, f)

    print("\n✓ Both models saved to models/ folder")

    # Return both models and feature names
    return {
        'random_forest': rf_model,
        'logistic_regression': lr_model,
        'features': features,
        'rf_accuracy': rf_accuracy,
        'lr_accuracy': lr_accuracy
    }


def analyze_openings_by_elo(df):
    """
    Core analysis: calculates win rates for each opening at different ELO levels.
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

    # Filter out openings with too few games (increased threshold for better reliability)
    opening_stats = opening_stats[opening_stats['total_games'] >= 100]

    return opening_stats


def main():
    """
    Main function that runs the entire project pipeline.
    """
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

        # Check for manually downloaded file
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
            print("\n" + "=" * 60)
            print("ERROR: No chess dataset found!")
            print("=" * 60)
            print("\nPlease download the chess dataset manually:")
            print("1. Go to: https://www.kaggle.com/datasets/shkarupylomaxim/chess-games-dataset-lichess-2017-may")
            print("2. Click 'Download' button")
            print("3. Extract the zip file")
            print("4. Place 'games_metadata_profile_2024_01.csv' in your 'data/' folder")
            print("=" * 60)
            return

    # Step 2: Load and clean data
    print("\nStep 1: Loading data...")
    df = load_and_clean_data(DATA_PATH)
    print(f"Loaded {len(df):,} games")

    # Step 3: Extract openings
    print("\nStep 2: Extracting openings...")
    df = extract_openings(df)
    print(f"Found {df['opening_name'].nunique():,} unique openings")

    # Step 4: Train ML models (
    print("\nStep 3: Training models...")
    model_results = train_model(df)

    # Extract models and results
    rf_model = model_results['random_forest']
    lr_model = model_results['logistic_regression']
    features = model_results['features']

    # Step 5: Analyze openings by ELO
    print("\nStep 4: Analyzing openings...")
    opening_stats = analyze_openings_by_elo(df)
    print(f"Analyzing {len(opening_stats)} opening-category combinations")
    print(f"(Filtered to openings with ≥100 games for reliability)")

    # Step 6: Create visualizations
    print("\nStep 5: Creating visualizations...")
    plot_win_rates(opening_stats)
    plot_feature_importance(rf_model, features)

    # Step 7: Save results
    opening_stats.to_csv('results/opening_analysis.csv', index=False)
    print(f"\n✓ Analysis complete! Check the 'results/' folder for charts.")
    print(f"✓ Opening statistics saved to results/opening_analysis.csv")
    print(f"✓ Models saved to models/ folder")

    # Print top findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS (STATISTICALLY RELIABLE):")
    print("=" * 60)

    # Overall statistics
    total_games = opening_stats['total_games'].sum()
    avg_games = opening_stats['total_games'].mean()
    print(f"\nTotal games analyzed: {total_games:,}")
    print(f"Average games per opening: {avg_games:.0f}")

    # Overall white win rate
    overall_white_rate = (opening_stats['white_wins'].sum() /
                          opening_stats['total_games'].sum())
    print(f"Overall white win rate: {overall_white_rate:.1%}")
    print("(Expected in real chess: 48-55%)")

    # Filter for reliable findings (500+ games)
    reliable = opening_stats[opening_stats['total_games'] >= 500]

    if not reliable.empty:
        print(f"\nHighly reliable findings (≥500 games): {len(reliable)} openings")

        # Best reliable opening for each category
        for category in ['Beginner (<1200)', 'Intermediate (1400-1600)', 'Advanced (1600-2000)']:
            cat_reliable = reliable[reliable['elo_category'] == category]
            if not cat_reliable.empty:
                best = cat_reliable.loc[cat_reliable['white_win_rate'].idxmax()]
                print(f"\n{category}:")
                print(f"  Best opening: {best['opening_name']}")
                print(f"  White win rate: {best['white_win_rate']:.1%}")
                print(f"  Sample size: {int(best['total_games'])} games ✓ RELIABLE")

    # Most reliable findings (highest sample size)
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
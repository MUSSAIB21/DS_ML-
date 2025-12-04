#This file creates charts for the project, uses matplotlib and seaborn


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Set style for academic look
plt.style.use('default')
sns.set_palette("deep")


def plot_win_rates(opening_stats):
    #Creates a heatmap showing win rates by opening and ELO category
    print("Creating win rate heatmap...")

    # Pivot data for heatmap
    pivot_data = opening_stats.pivot(index='opening_name',
                                     columns='elo_category',
                                     values='white_win_rate')

    # Only keep openings with data across multiple categories
    pivot_data = pivot_data.dropna(thresh=2)

    plt.figure(figsize=(12, 8))

    # Create heatmap
    sns.heatmap(pivot_data,
                annot=True,
                fmt='.1%',
                cmap='RdYlGn',
                center=0.5,
                cbar_kws={'label': 'White Win Rate'})

    plt.title('Chess Opening Win Rates by Player Rating', fontsize=16, fontweight='bold')
    plt.xlabel('ELO Rating Category', fontsize=12)
    plt.ylabel('Opening Name', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Save chart
    filepath = 'results/opening_win_rates_heatmap.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {filepath}")
    plt.close()


def plot_feature_importance(model, features):
    """
    Plots feature importance from the Random Forest model.
    Shows which factors matter most for predicting wins.
    """
    print("Creating feature importance chart...")

    # Get feature importance
    importances = model.feature_importances_

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 6))

    # Horizontal bar plot
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Feature Importance for Predicting Chess Game Outcomes',
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    # Add value labels
    for index, value in enumerate(importance_df['importance']):
        plt.text(value, index, f'{value:.3f}', va='center')

    plt.tight_layout()

    # Save chart
    filepath = 'results/feature_importance.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved feature importance to {filepath}")
    plt.close()


def plot_opening_popularity(df):

    #Shows the most popular openings across all games.

    print("Creating opening popularity chart...")

    # Count games per opening
    opening_counts = df['opening_name'].value_counts().head(10)

    plt.figure(figsize=(12, 6))

    # Bar plot
    bars = plt.bar(range(len(opening_counts)), opening_counts.values)

    # Color bars by win rate
    for i, bar in enumerate(bars):
        # Simple gradient coloring
        bar.set_color(plt.cm.viridis(i / len(bars)))

    plt.xticks(range(len(opening_counts)), opening_counts.index, rotation=45, ha='right')
    plt.ylabel('Number of Games')
    plt.title('Top 10 Most Popular Chess Openings', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save chart
    filepath = 'results/opening_popularity.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved popularity chart to {filepath}")
    plt.close()


def create_summary_report(opening_stats, df):

    #Creates a simple text-based report.

    report = []
    report.append("CHESS OPENINGS ANALYSIS - SUMMARY REPORT")
    report.append("=" * 50)
    report.append(f"Total games analyzed: {len(df):,}")
    report.append(f"Unique openings: {df['opening_name'].nunique()}")
    report.append(f"ELO range: {df['avg_elo'].min():.0f} - {df['avg_elo'].max():.0f}")
    report.append("")

    #Best opening overall
    overall_best = opening_stats.groupby('opening_name')['white_win_rate'].mean().idxmax()
    report.append(f"Best opening overall: {overall_best}")
    report.append("")

    #Best by category
    report.append("Best openings by ELO category:")
    for category in opening_stats['elo_category'].unique():
        subset = opening_stats[opening_stats['elo_category'] == category]
        if not subset.empty:
            best = subset.loc[subset['white_win_rate'].idxmax()]
            report.append(f"  {category}: {best['opening_name']} ({best['white_win_rate']:.1%} win rate)")

    # Saves report
    with open('results/summary_report.txt', 'w') as f:
        f.write('\n'.join(report))

    print("Saved summary report to results/summary_report.txt")

if __name__ == "__main__":
    #Creating sample data for testing the visualizations
    print("Creating sample visualizations...")

    sample_stats = pd.DataFrame({
        'elo_category': ['Beginner', 'Beginner', 'Advanced', 'Advanced'],
        'opening_name': ['Sicilian', 'Italian', 'Sicilian', 'Italian'],
        'white_win_rate': [0.45, 0.52, 0.51, 0.48],
        'total_games': [1000, 1500, 800, 600]
    })

    plot_win_rates(sample_stats)
    print("Sample visualizations created!")
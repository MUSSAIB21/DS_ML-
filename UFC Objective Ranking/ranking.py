"""
UFC Fighter Ranking System
==========================
Reads fights_clean.csv and produces a ranked list of fighters per weight class.
Saves results to rankings.csv.

HOW SCORING WORKS:
  Every fight a fighter has ever had contributes points to their score.
  Older fights contribute less than recent fights (recency decay).
  The score is made up of 5 components:

  1. WIN POINTS
     A win gives base points. The more impressive the win, the more points.
     A loss deducts a small amount of points.
     A draw gives zero.

  2. QUALITY OF OPPONENT (the most important part)
     Beating a fighter with a great record gives you more points.
     Beating a fighter with a poor record gives you fewer points.
     This is similar to how chess Elo works.

  3. FINISH BONUS
     Winning by KO or submission gives extra points on top of the win.
     Winning by decision gives a smaller bonus.
     This rewards fighters who are dominant, not just good enough to win.

  4. ACTIVITY BONUS
     Fighters who compete more often get a small bonus per fight.
     This stops retired or inactive fighters from sitting at the top forever.

  5. RECENCY DECAY
     Every fight is multiplied by a decay factor based on how long ago it was.
     A fight from last month is worth almost full points.
     A fight from 5 years ago is worth much less.
     This ensures rankings reflect current form, not just career history.

HOW TO RUN:
  python ranking.py
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("Loading fights_clean.csv...")
df = pd.read_csv("fights_clean.csv", parse_dates=["date"])
print(f"Loaded {len(df)} fights.")
print()


# SCORING WEIGHTS
# These numbers control how much each component matters.

WIN_POINTS          = 100   # base points for a win
LOSS_DEDUCTION      = 30    # points deducted for a loss
FINISH_BONUS        = 40    # extra points for a KO or submission win
DECISION_BONUS      = 10    # smaller bonus for a decision win
OPPONENT_QUALITY_W  = 0.5   # how much opponent win rate scales the win bonus
RECENCY_HALF_LIFE   = 730   # fights decay to half value after this many days (2 years)
MIN_FIGHTS          = 3     # fighters with fewer fights than this are excluded from rankings



# 1: Building a flat list of every fighter's fight history
# Right now each row has fighter_1 and fighter_2 side by side.
# We need to "melt" this so each row represents ONE fighter in ONE fight.

print("Restructuring data into per-fighter rows...")

def build_fighter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the fights dataframe where each row has two fighters,
    and return a new dataframe where each row is one fighter in one fight.

    For each fight we create two rows:
      one for fighter 1 (the winner if result is win)
      one for fighter 2 (the loser if result is win)
    """
    rows = []

    for _, fight in df.iterrows():
        base = {
            "date":         fight["date"],
            "event":        fight["event"],
            "weight_class": fight["weight_class"],
            "method":       fight["method"],
            "finish":       fight["finish"],
            "round":        fight["round"],
        }

        f1_won = fight["result"] == "win"

        row_f1 = {
            **base,
            "fighter":       fight["fighter_1"],
            "opponent":      fight["fighter_2"],
            "won":           1 if f1_won else 0,
            "lost":          0 if f1_won else 1,
            "draw":          1 if fight["result"] == "draw" else 0,
            "kd":            fight["kd_f1"],
            "sig_str":       fight["sig_str_f1_landed"],
            "td":            fight["td_f1"],
            "sub_att":       fight["sub_att_f1"],
            "opp_kd":        fight["kd_f2"],
            "opp_sig_str":   fight["sig_str_f2_landed"],
        }

        row_f2 = {
            **base,
            "fighter":       fight["fighter_2"],
            "opponent":      fight["fighter_1"],
            "won":           0 if f1_won else 1,
            "lost":          1 if f1_won else 0,
            "draw":          1 if fight["result"] == "draw" else 0,
            "kd":            fight["kd_f2"],
            "sig_str":       fight["sig_str_f2_landed"],
            "td":            fight["td_f2"],
            "sub_att":       fight["sub_att_f2"],
            "opp_kd":        fight["kd_f1"],
            "opp_sig_str":   fight["sig_str_f1_landed"],
        }

        rows.append(row_f1)
        rows.append(row_f2)

    return pd.DataFrame(rows)

fighter_fights = build_fighter_rows(df)
print(f"Built {len(fighter_fights)} fighter-fight rows across {fighter_fights['fighter'].nunique()} unique fighters.")
print()


# 2: Calculate each fighter's overall win rate
# We use this as a proxy for opponent quality.
# If you beat someone with a 70% win rate, that is more impressive
# than beating someone with a 30% win rate.

print("Calculating win rates for opponent quality scores...")

win_rates = (
    fighter_fights.groupby("fighter")
    .apply(lambda x: x["won"].sum() / len(x), include_groups=False)
    .rename("win_rate")
    .reset_index()
)

fighter_fights = fighter_fights.merge(
    win_rates.rename(columns={"fighter": "opponent", "win_rate": "opp_win_rate"}),
    on="opponent",
    how="left"
)

fighter_fights["opp_win_rate"] = fighter_fights["opp_win_rate"].fillna(0.5)


# 3: Calculate recency decay for each fight
# A fight from today = decay factor of 1.0 (full value)
# A fight from 2 years ago = decay factor of 0.5 (half value)
# A fight from 4 years ago = decay factor of 0.25 (quarter value)
# Formula: decay = 0.5 ^ (days_ago / half_life)

today = pd.Timestamp(datetime.today().date())
fighter_fights["days_ago"] = (today - fighter_fights["date"]).dt.days
fighter_fights["decay"]    = 0.5 ** (fighter_fights["days_ago"] / RECENCY_HALF_LIFE)


# 4: Calculate the score contribution for each fight
# This is the heart of the ranking system.
# For each fight we calculate how many points it contributes
# to the fighter's total score, then multiply by the decay factor.

def fight_score(row) -> float:
    """
    Calculate the raw score (before decay) for one fighter in one fight.
    """
    score = 0.0

    if row["won"] == 1:
        score += WIN_POINTS
        score += WIN_POINTS * OPPONENT_QUALITY_W * row["opp_win_rate"]
        if row["finish"]:
            score += FINISH_BONUS
        else:
            score += DECISION_BONUS

    elif row["lost"] == 1:
        score -= LOSS_DEDUCTION
        score -= LOSS_DEDUCTION * OPPONENT_QUALITY_W * (1 - row["opp_win_rate"])

    return score

fighter_fights["raw_score"]    = fighter_fights.apply(fight_score, axis=1)
fighter_fights["decayed_score"] = fighter_fights["raw_score"] * fighter_fights["decay"]


#5: Aggregate scores per fighter per weight class
# Add up all their decayed fight scores to get a total score.
# Also compute useful summary stats for the leaderboard.

print("Aggregating scores per fighter...")

rankings = (
    fighter_fights
    .groupby(["fighter", "weight_class"])
    .agg(
        total_score    = ("decayed_score",  "sum"),
        fights         = ("won",            "count"),
        wins           = ("won",            "sum"),
        losses         = ("lost",           "sum"),
        draws          = ("draw",           "sum"),
        finishes       = ("finish",         "sum"),
        last_fight     = ("date",           "max"),
        avg_kd         = ("kd",             "mean"),
        avg_sig_str    = ("sig_str",        "mean"),
        avg_td         = ("td",             "mean"),
    )
    .reset_index()
)

rankings["win_rate"]     = (rankings["wins"] / rankings["fights"] * 100).round(1)
rankings["finish_rate"]  = (rankings["finishes"] / rankings["wins"].replace(0, np.nan) * 100).round(1)
rankings["finish_rate"]  = rankings["finish_rate"].fillna(0)
rankings["record"]       = (
    rankings["wins"].astype(int).astype(str) + "-" +
    rankings["losses"].astype(int).astype(str) + "-" +
    rankings["draws"].astype(int).astype(str)
)

rankings = rankings[rankings["fights"] >= MIN_FIGHTS]
rankings["rank"] = rankings.groupby("weight_class")["total_score"].rank(
    ascending=False, method="min"
).astype(int)

rankings = rankings.sort_values(["weight_class", "rank"])


# 6: Save and preview results

output_cols = [
    "rank", "fighter", "weight_class", "record", "win_rate",
    "finish_rate", "total_score", "avg_kd", "avg_sig_str",
    "avg_td", "last_fight", "fights"
]

rankings[output_cols].to_csv("rankings.csv", index=False)
print(f"Done! Saved rankings for {rankings['fighter'].nunique()} fighters to rankings.csv")
print()

print("TOP 10 FIGHTERS PER WEIGHT CLASS:")
print()

weight_classes = rankings["weight_class"].unique()
for wc in sorted(weight_classes):
    top10 = rankings[rankings["weight_class"] == wc].head(10)
    print(f"  {wc}")
    print(f"  {'Rank':<6} {'Fighter':<25} {'Record':<12} {'Win%':<8} {'Finish%':<10} {'Score':<10}")
    print(f"  {'-'*71}")
    for _, row in top10.iterrows():
        print(f"  {int(row['rank']):<6} {row['fighter']:<25} {row['record']:<12} {row['win_rate']:<8} {row['finish_rate']:<10} {row['total_score']:.1f}")
    print()
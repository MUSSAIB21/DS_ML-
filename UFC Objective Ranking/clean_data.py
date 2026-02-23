"""
Data Cleaning Script
====================
Reads fights.csv and produces a cleaned version called fights_clean.csv.

What this script does step by step:
  1. Remove future fights (no results yet)
  2. Convert the date column to a real date
  3. Split the strikes columns into "landed" and "attempted"
  4. Convert all stat columns to numbers
  5. Filter out Open Weight and non-standard bouts
  6. Add a clean "result" column (win, draw, no contest)
  7. Save to fights_clean.csv

HOW TO RUN:
  pip install pandas
  python clean_data.py
"""

import pandas as pd

print("Loading fights.csv...")
df = pd.read_csv("fights.csv")
print(f"Loaded {len(df)} rows.")
print()



# Removing future fights
# Future fights have no method, round, or time filled in.
# We keep only rows where method is not empty.

before = len(df)
df = df[df["method"].notna() & (df["method"].str.strip() != "")]
print(f"Step 1: Removed {before - len(df)} future/incomplete fights. {len(df)} remain.")


# Convert date to a real date column
# "February 28, 2026" becomes a proper date object
# so we can sort by date and calculate recency later.

df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")
bad_dates = df["date"].isna().sum()
if bad_dates:
    print(f"Step 2: Warning, {bad_dates} rows had unparseable dates and will be dropped.")
    df = df[df["date"].notna()]
print(f"Step 2: Dates converted. Date range: {df['date'].min().date()} to {df['date'].max().date()}")


# Split strikes columns into landed and attempted
# The raw data looks like "110 of 203"
# We split this into sig_str_landed_f1 and sig_str_att_f1
# We do this for both fighter 1 and fighter 2.

def split_strikes(series: pd.Series, prefix: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a column like "110 of 203" and splits it into two new columns:
      prefix_landed  = 110
      prefix_att     = 203
    If the value is just a plain number (some older fights), landed = that number, att = NaN.
    """
    split = series.str.extract(r"(\d+)\s+of\s+(\d+)")
    df[f"{prefix}_landed"] = pd.to_numeric(split[0], errors="coerce")
    df[f"{prefix}_att"]    = pd.to_numeric(split[1], errors="coerce")

    mask = df[f"{prefix}_landed"].isna()
    df.loc[mask, f"{prefix}_landed"] = pd.to_numeric(series[mask], errors="coerce")

    return df

df = split_strikes(df["sig_str_f1"], "sig_str_f1", df)
df = split_strikes(df["sig_str_f2"], "sig_str_f2", df)

df = df.drop(columns=["sig_str_f1", "sig_str_f2"])
print(f"Step 3: Strikes columns split into landed and attempted.")


# Convert stat columns to numbers
# Columns like kd, td, sub_att come in as text strings.
# We convert them all to integers so we can do maths on them.

numeric_cols = ["kd_f1", "kd_f2", "td_f1", "td_f2", "sub_att_f1", "sub_att_f2", "round"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"Step 4: Converted {numeric_cols} to numeric.")


# Filter out non-standard weight classes
# Early UFC used Open Weight and some other odd classes.
# These arent useful for a modern ranking system.

standard_classes = [
    "Heavyweight",
    "Light Heavyweight",
    "Middleweight",
    "Welterweight",
    "Lightweight",
    "Featherweight",
    "Bantamweight",
    "Flyweight",
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
]

before = len(df)
df = df[df["weight_class"].isin(standard_classes)]
print(f"Step 5: Removed {before - len(df)} non-standard weight class fights. {len(df)} remain.")


# Add a clean result column
# Right now "winner" is either a fighter name, "Draw / No Contest", or blank.
# We add a simpler column:
#   "win"        = fighter 1 won
#   "draw"       = draw
#   "no contest" = no contest

def classify_result(row):
    winner = str(row["winner"]).strip().lower()
    if winner in ("draw / no contest", "draw", "no contest", "nc"):
        if "draw" in winner:
            return "draw"
        return "no contest"
    if row["winner"] == row["fighter_1"]:
        return "win"
    return "unknown"

df["result"] = df.apply(classify_result, axis=1)
print(f"Step 6: Result breakdown:")
print(df["result"].value_counts().to_string())


# Add a finish column
# Tells us whether the fight ended by finish (KO/TKO or SUB)
# or went to the judges (decision).
# Useful later for calculating finish rate in the ranking system.

df["finish"] = df["method"].str.upper().str.startswith(("KO", "SUB", "TKO"))
print(f"Step 7: Finish rate across all fights: {df['finish'].mean():.1%}")


# FINAL: Save cleaned data

df.to_csv("fights_clean.csv", index=False)
print()
print(f"Done! Saved {len(df)} cleaned fights to fights_clean.csv")
print()
print("Column overview:")
print(df.dtypes.to_string())
print()
print("Preview:")
print(df.head(5).to_string())
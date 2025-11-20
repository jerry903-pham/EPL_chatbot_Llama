import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROLLING_WINDOW = 3


def calculate_god_mode_features(group):
    """
    Calculates complex metrics:
    1. Rest Days (Fatigue)
    2. Aggression Score (Risk of Red Card)
    3. Clinical Efficiency (Goals per Shot on Target)
    """
    # Shift data so we don't see the future (Stats from PREVIOUS games)
    shifted = group.shift(1)

    # A. Rolling Basics
    cols = ['Shots', 'ShotsTarget', 'Goals', 'Corners', 'Fouls', 'Yellows', 'Reds']
    rolling = shifted[cols].rolling(ROLLING_WINDOW, min_periods=1).mean()

    # B. Derived "God Mode" Metrics

    # Aggression: Weighted score (Foul=1, Yellow=10, Red=25)
    aggression = (rolling['Fouls'] * 1) + (rolling['Yellows'] * 10) + (rolling['Reds'] * 25)

    # Efficiency: Goals / Shots on Target (Avoid divide by zero)
    efficiency = rolling['Goals'] / (rolling['ShotsTarget'] + 0.1)

    # Rest Days: Calculate days since previous match
    # We use the group's actual dates (unshifted) to find the gap
    rest_days = group['Date'].diff().dt.days.fillna(7).shift(1)

    result = pd.DataFrame({
        'Date': group['Date'],  # Keep Date for merging
        'Shots_Roll': rolling['Shots'],
        'ST_Roll': rolling['ShotsTarget'],
        'Goals_Roll': rolling['Goals'],
        'Aggression_Roll': aggression,
        'Efficiency_Roll': efficiency,
        'Rest_Days_Roll': rest_days
    })

    return result


print("INITIALIZING GOD MODE SEQUENCE...")
print("Loading Data...")
df = pd.read_csv("epl_matches.csv")

# CRITICAL: Date Handling
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.sort_values('Date')

# ==========================================
# 2. GLOBAL METRICS (Referees & Market)
# ==========================================

# --- A. Market Variance (Uncertainty) ---
df['Implied_Prob_H'] = 1 / df['AvgH']
df['Implied_Prob_A'] = 1 / df['AvgA']
# High MaxH - AvgH means bookies disagree -> Volatility
df['Odds_Variance'] = df['MaxH'] - df['AvgH']

#B. Referee Profiling ---
print("Profiling Referees...")
# Calculate average "Strictness" per referee (Cards + Fouls)
ref_stats = df.groupby('Referee')[['HF', 'AF', 'HY', 'AY', 'HR', 'AR']].mean()
ref_stats['Strictness_Score'] = (ref_stats['HF'] + ref_stats['AF']) + \
                                (ref_stats['HY'] + ref_stats['AY']) * 3 + \
                                (ref_stats['HR'] + ref_stats['AR']) * 10
df = df.merge(ref_stats['Strictness_Score'], on='Referee', how='left')

# 3. TEAM PERFORMANCE METRICS
print("Calculating Physiology & Psychology Stats...")

# 1. Stack Home and Away data to create a single timeline per team
home_data = df[['Date', 'HomeTeam', 'HS', 'HST', 'FTHG', 'HC', 'HF', 'HY', 'HR']].rename(
    columns={'HomeTeam': 'Team', 'HS': 'Shots', 'HST': 'ShotsTarget', 'FTHG': 'Goals', 'HC': 'Corners', 'HF': 'Fouls',
             'HY': 'Yellows', 'HR': 'Reds'}
)
away_data = df[['Date', 'AwayTeam', 'AS', 'AST', 'FTAG', 'AC', 'AF', 'AY', 'AR']].rename(
    columns={'AwayTeam': 'Team', 'AS': 'Shots', 'AST': 'ShotsTarget', 'FTAG': 'Goals', 'AC': 'Corners', 'AF': 'Fouls',
             'AY': 'Yellows', 'AR': 'Reds'}
)

all_stats = pd.concat([home_data, away_data]).sort_values(['Team', 'Date'])

# 2. Apply the "God Mode" calculations
# Fix: include_groups=False to silence warning
advanced_stats = all_stats.groupby('Team').apply(calculate_god_mode_features, include_groups=False).reset_index(level=0)

# 3. Prepare specific DataFrames for Home and Away merges to avoid Rename/Suffix bugs
home_stats_to_merge = advanced_stats.rename(columns={
    'Team': 'HomeTeam',
    'Shots_Roll': 'H_Shots', 'ST_Roll': 'H_ST', 'Goals_Roll': 'H_Goals',
    'Aggression_Roll': 'H_Aggression', 'Efficiency_Roll': 'H_Efficiency', 'Rest_Days_Roll': 'H_Rest'
})

away_stats_to_merge = advanced_stats.rename(columns={
    'Team': 'AwayTeam',
    'Shots_Roll': 'A_Shots', 'ST_Roll': 'A_ST', 'Goals_Roll': 'A_Goals',
    'Aggression_Roll': 'A_Aggression', 'Efficiency_Roll': 'A_Efficiency', 'Rest_Days_Roll': 'A_Rest'
})

# 4. Merge back to main DataFrame
# Merge Home
df = pd.merge(df, home_stats_to_merge, on=['Date', 'HomeTeam'], how='left')

# Merge Away
df = pd.merge(df, away_stats_to_merge, on=['Date', 'AwayTeam'], how='left')

# Fill NaNs (First games of season have no history)
df = df.fillna(0)

# 4. ENCODING & TRAINING
le_team = LabelEncoder()
le_ref = LabelEncoder()
le_res = LabelEncoder()

# Fit Encoders
all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
le_team.fit(all_teams)
le_ref.fit(df['Referee'])
df['Target'] = le_res.fit_transform(df['FTR'])

df['Home_Code'] = le_team.transform(df['HomeTeam'])
df['Away_Code'] = le_team.transform(df['AwayTeam'])
df['Ref_Code'] = le_ref.transform(df['Referee'])

# Define the "God Mode" Feature Set
predictors = [
    'Home_Code', 'Away_Code', 'Ref_Code', 'Strictness_Score',
    'AvgH', 'AvgD', 'AvgA', 'Odds_Variance',
    'H_Shots', 'H_ST', 'H_Goals', 'H_Aggression', 'H_Efficiency', 'H_Rest',
    'A_Shots', 'A_ST', 'A_Goals', 'A_Aggression', 'A_Efficiency', 'A_Rest'
]

X = df[predictors]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training RandomForest on {len(predictors)} advanced features...")
model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# 5. SAVING THE ARTIFACTS
with open("model_package.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "encoders": {'team': le_team, 'ref': le_ref, 'res': le_res},
        "predictors": predictors,
        "ref_db": ref_stats,  # To lookup referee strictness
        "team_db": advanced_stats,  # To lookup team form
        "raw_db": all_stats  # To lookup dates for rest calculation
    }, f)

print("saved to 'model_package.pkl'")

# Feature Importance Chart
try:
    plt.figure(figsize=(12, 8))
    importances = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)
    sns.barplot(x=importances, y=importances.index, palette='magma')
    plt.title("God Mode: Factor Importance")
    plt.xlabel("Impact on Prediction")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Visualization skipped (Check graphical environment)")
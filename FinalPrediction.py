# FinalPrediction.py
# - Loads ATP matches (Jeff Sackmann's dataset) from a CSV file
# - Builds a player-level dataset (winner + loser rows)
# - Labels "injury" when score has RET/W/O (loser only)
# - Adds a few basic workload features
# - One-hot encodes a few categorical columns with pandas.get_dummies
# - Plots distributions using histograms, KDE and boxplots, simple category counts 
# - Trains three models this time (ExtraTrees, LightGBM, NuSVC) and evaluates them
# - Will need to make confusion matrices, ROC (TPR VS FPR) curves,
#   and an inverted ROC (TNR vs FNR) to visualize "negative" rates 

# Comments included in the code to make each step more understandable

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from platform import python_version

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import NuSVC
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report ,confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score,  precision_recall_curve,average_precision_score

# Configuration / constants (easy to edit)
CSV_FILEPATH = 'C:/Users/kleon/OneDrive/Documents/Dissertation/Dissertation-Repository/Datasets/atp_matches_2019.csv'
RANDOM_STATE = 42
ROLL_MINUTES_WINDOW = 5  # window for rolling mean of minutes played
ROLL_INJURIES_WINDOW = 10  # window for rolling sum of previous injuries
TEST_SIZE = 0.2  # 80/20 train/test split

sns.set(style="whitegrid")

# Libraries and Python version
libraries = {
    "Pandas": pd,
    "Matplotlib": matplotlib,
    "Seaborn": sns,
    "NumPy": np,
    "Scikit-Learn": sklearn,
}

# Libraries version
print("Library Version:\n")
print(f"{'':-^20} | {'':-^10}")
print(f"{'Library':^20} | {'Version':^10}")
print(f"{'':-^20} | {'':-^10}")

for lib_name, lib_mod in sorted(libraries.items()):
    print(f"{lib_name:<20} | {lib_mod.__version__:>10}")

# Helper functions

def load_matches(path):
    """
    Load ATP matches from a CSV file and converts tourney_date to a date time (YYYYMMDD -> Timestamp).
    This enables time-based features like days since last match. 
    """
    # errors="coerce" set bad values to NaT or NaN 
    df = pd.read_csv(path)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    return df

def label_injuries_from_scores(matches):
    """
    Basic injury heuristic:
        - If score contains 'RET' or 'W/O', assume the loser retired/withdrew -> injury event.
        Returns a 0/1 Series aligned to matches rows.
    """
    score_str = matches["score"].astype(str)
    return score_str.str.contains(r"RET|W/O", case=False, na=False).astype(int)

def make_player_rows(df, injury_loser, side):
    """
    Create a player-centric row for either 'winner' or 'loser' of the match.
    We map the right columns using a prefix (w_ for winner, l_ for loser).
    Only the losing side gets Injury=1 when RET/W/O is detected; winner gets 0.
    """
    assert side in ("winner", "loser")
    prefix = "w_" if side == "winner" else "l_"

    # Column name mapping based on perspective
    name_col = "winner_name" if side == "winner" else "loser_name"
    opp_col = "loser_name" if side == "winner" else "winner_name"
    age_col = "winner_age" if side == "winner" else "loser_age"
    ht_col = "winner_ht" if side == "winner" else "loser_ht"
    rank_col = "winner_rank" if side == "winner" else "loser_rank"
    rank_pts_col = "winner_rank_points" if side == "winner" else "loser_rank_points"

    # Numeric safe conversions for match stats and bio/rank info
    out = pd.DataFrame({
        "player_name": df[name_col],
        "opponent_name": df[opp_col],
        "date": df["tourney_date"],
        "tourney_name": df["tourney_name"],
        "surface": df["surface"],
        "round": df["round"],
        "minutes": pd.to_numeric(df["minutes"], errors="coerce"),
        "aces": pd.to_numeric(df[prefix + "ace"], errors="coerce"),
        "double_faults": pd.to_numeric(df[prefix + "df"], errors="coerce"),
        "serve_points": pd.to_numeric(df[prefix + "svpt"], errors="coerce"),
        "first_in": pd.to_numeric(df[prefix + "1stIn"], errors="coerce"),
        "first_won": pd.to_numeric(df[prefix + "1stWon"], errors="coerce"),
        "second_won": pd.to_numeric(df[prefix + "2ndWon"], errors="coerce"),
        "serve_games": pd.to_numeric(df[prefix + "SvGms"], errors="coerce"),
        "bp_saved": pd.to_numeric(df[prefix + "bpSaved"], errors="coerce"),
        "bp_faced": pd.to_numeric(df[prefix + "bpFaced"], errors="coerce"),
        "rank": pd.to_numeric(df[rank_col], errors="coerce"),
        "rank_points": pd.to_numeric(df[rank_pts_col], errors="coerce"),
        "age": pd.to_numeric(df[age_col], errors="coerce"),
        "height_cm": pd.to_numeric(df[ht_col], errors="coerce"),
        "best_of": pd.to_numeric(df["best_of"], errors="coerce"),
        "score": df["score"].astype(str),
    })

    # Injury label: losers can be 1 if RET/W/O; winner always 0
    out["Injury"] = injury_loser.values if side == "loser" else 0
    return out

def add_player_workload_features(players_df):
    """
    Add simple workload features per player:
    - Recovery_Time: days since last match (per player)
    - Training_Intensity: rolling mean of minutes (per player, excluding current match)
    - Previous_Injuries: rolling sum of past injuries (per player, excluding current match)
    Missing values are filled with medians for stability
    """
    # Days since last match
    players_df["Recovery_Time"] = players_df.groupby("player_name")["date"].diff().dt.days

    # Rolling mean of minutes, using previous matches only
    players_df["Training_Intensity"] = (
        players_df.groupby("player_name")["minutes"]
        .transform(lambda s: s.shift(1).rolling(ROLL_MINUTES_WINDOW, min_periods=1).mean())
    )

    # Rolling sum of prior injuries, using previous matches only
    players_df["Previous_Injuries"] = (
        players_df.groupby("player_name")["Injury"]
        .transform(lambda s: s.shift(1).rolling(ROLL_INJURIES_WINDOW, min_periods=1).sum())
    )

    # Fill NaNs with column medians
    for col in ["Recovery_Time", "Training_Intensity", "Previous_Injuries"]:
        players_df[col] = players_df[col].fillna(players_df[col].median())

    return players_df

def plot_feature_distributions(df, col, target_col="Injury"):
    """
    Quick visualization helper: histogram + KDE + boxplot for a given column
    KDE tries to split by target (Injury) if available.
    """
    if col not in df.columns:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df[col], bins=20, kde=True, color="skyblue", ax=axes[0], edgecolor="black")
    axes[0].set_title(f"{col} Histogram")
    try:
        sns.kdeplot(data=df, x=col, hue=target_col, fill=True, ax=axes[1], palette={0: "green", 1: "red"})
    except Exception:
        sns.kdeplot(df[col], fill=True, color="skyblue", ax=axes[1])
    axes[1].set_title(f"{col} Density")
    sns.boxplot(x=df[col], ax=axes[2], color="skyblue")
    axes[2].set_title(f"{col} Boxplot")
    plt.tight_layout()
    plt.show()

def plot_counts_by_category_and_label(df, column, hue="Injury", order=None, y_limit=None):
    """
    Simple count plot for a categorical column split by target label (Injury).
    Adds labels on top of bars to show counts
    """
    if column not in df.columns:
        return 
    plt.figure(figsize=(8,4.5))
    ax = sns.countplot(data=df, x=column, hue=hue, order=order, palette={0:"green", 1:"red"})
    ax.set_title(f"{column} x {hue}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    if y_limit is not None:
        ax.set_ylim(top=y_limit)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(str(int(height)), (p.get_x() + p.get_width() / 2.0, height),
                    ha="center", va="bottom", xytext=(0,3),textcoords="offset points")
    plt.tight_layout()
    plt.show()


# Main flow

matches = load_matches(CSV_FILEPATH)
print("Matches loaded:", matches.shape)
print("\nMatches preview:")
print(matches.head())

injury_loser = label_injuries_from_scores(matches)

# Build player-level rows and sort for rolling features 
winners = make_player_rows(matches, injury_loser, side="winner")
losers = make_player_rows(matches, injury_loser, side="loser")
players = pd.concat([winners, losers], axis=0, ignore_index=True)
players = players.sort_values(["player_name", "date"]).reset_index(drop=True)

print("\nPlayer=level dataset preview:")
print(players.head())

#Quick info table (dtype, unique, null)
df_info = pd.DataFrame(players.dtypes, columns=["Dtype"])
df_info["Unique"] = players.nunique().values 
df_info["Null"] = players.isnull().sum().values
print("\nplayers info (dtype/unique/null):")
print(df_info.head(20))

#Workload features 
players = add_player_workload_features(players)

#Feature distributions
for c in ["Training_Intensity", "Recovery_Time", "Previous_Injuries", "minutes"]:
    plot_feature_distributions(players, c, target_col="Injury")

# Adding Injury Pie Chart
counts = players["Injury"].value_counts()
plt.figure(figsize=(4.2,4.3))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140,
        colors=["#1a7009", "#af0c0c"])
plt.title("Injury (0/1) Distribution")
plt.axis("equal")
plt.show()

# Simple category counts 
plot_counts_by_category_and_label(players, "surface", hue="Injury")
plot_counts_by_category_and_label(players, "round", hue="Injury")

# Encode categoricals with get_dummies 
category_columns = [c for c in ["surface", "round", "tourney_name"] if c in players.columns]
df = pd.get_dummies(players, columns=category_columns, dummy_na=True)

# Assemble features X and target y
feature_cols = [
    "minutes", "Training_Intensity", "Recovery_Time", "Previous_Injuries",
    "aces", "double_faults", "serve_points", "first_in", "first_won",
    "second_won", "serve_games", "bp_saved", "bp_faced", "rank", "rank_points", "age", "height_cm",
]

feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df["Injury"].astype(int)

print("\nTarget distribution (raw):")
print(y.value_counts())

# If no positive/ negative variation, stop
if y.nunique() < 2:
    print("\n[Warning] Target is single-class (no variation). Skipping model training")
    raise SystemExit(0)

# Train/test split
stratify_arg = y if y.nunique() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_arg)


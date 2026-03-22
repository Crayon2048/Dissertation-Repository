# FinalPrediction.py
# - Loads ATP matches (Jeff Sackmann's dataset) from multiple yearly CSV fileS
# - Builds a player-level dataset (winner + loser rows)
# - Labels "injury" when score has RET/W/O (loser only)
# - Adds a few basic workload features and using shift(1) to prevent data leakage 
# - One-hot encodes a few categorical columns with pandas.get_dummies
# - Uses a time-based train/test split (train on 2017-2018, test on 2019)
#   so that the model is trained only on past data and tested on genuinely 
#   future matches making it at least a true prospective injury prediction
# - Plots distributions using histograms, KDE and boxplots, simple category counts 
# - Trains three models this time (ExtraTrees, LightGBM, NuSVC) and evaluates them
# - Adding confusion matrices, ROC (TPR VS FPR) curves, and Precision-Recall curve

# Comments included in the code to make each step more understandable

import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from platform import python_version
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import NuSVC
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score, 
    classification_report ,
    confusion_matrix,
    roc_curve, 
    roc_auc_score,  
    precision_recall_curve,
    average_precision_score,
)

# Configuration / constants (easy to edit)

# Folder containing all yearly CSV files (e.g. atp_matches_2017, atp_matches_2018.csv etc.)
# The glob patten below will load all matching files in that folder automatically
DATASET_FOLDER = r'C:/Users/kleon/OneDrive/Documents/Dissertation/Dissertation-Repository/Datasets/'

# Training years: models will learn from these years only
TRAIN_YEARS = [2021, 2022]

# Test year: the "future" year the model has never seen during training 
TEST_YEAR = 2023

RANDOM_STATE = 42
ROLL_MINUTES_WINDOW = 5  # window for rolling mean of minutes played
ROLL_INJURIES_WINDOW = 10  # window for rolling sum of previous injuries


sns.set(style="whitegrid")

# Libraries and Python version
libraries = {
    "Pandas": pd,
    "Matplotlib": matplotlib,
    "Seaborn": sns,
    "NumPy": np,
}

# Libraries version
print("Library Version:\n")
print(f"{'':-^20} | {'':-^10}")
print(f"{'Library':^20} | {'Version':^10}")
print(f"{'':-^20} | {'':-^10}")

for lib_name, lib_mod in sorted(libraries.items()):
    print(f"{lib_name:<20} | {lib_mod.__version__:>10}")

# Python Version
print()
print(f"Python Version: {python_version()}")

# Helper functions

def load_all_matches(folder, years):
    """
    Load multiple yearly ATP match CSV files and combine them into one DataFrame.
    Only loads files for the specified years so we have full control over 
    what goes into training vs testing.
    Each file should be named atp_matches_YYYY.csv (Jeff Sackmann format).
    tourney_data is converted from YYYYMMDD integer to a proper datetime.
    """
    all_dfs = []
    for year in years:
        pattern = f"{folder}atp_matches_{year}.csv"
        files = glob.glob(pattern)
        if not files:
            print(f"[Warning] No file found for year {year} at: {pattern}")
            continue
        for f in files:
            # errors="coerce" set bad values to NaT or NaN 
            df = pd.read_csv(f)
            df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
            all_dfs.append(df)
            print(f"Loaded: {f} ({len(df)} matches)")
    
    if not all_dfs:
        raise FileNotFoundError(f"No CSV files found in {folder} for years {years}")

    combined = pd.concat(all_dfs,axis=0, ignore_index=True)
    print(f"\nTotal matches loaded: {combined.shape[0]} rows across {len(all_dfs)} files(s)")
    return combined


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
    A 'side' column is added so we can filter to losers only later for 
    match-level evaluation 
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
        # Track which side this row came from so we can filter later
        "side": side,
    })

    # Injury label: losers can be 1 if RET/W/O; winner always 0
    out["Injury"] = injury_loser.values if side == "loser" else 0
    return out

def add_workload_features(players_df):
    """
    Add simple workload features per player using only past match data (shift(1) prevents leakage):
    - Recovery_Time: days since last match
    - Training_Intensity: rolling mean of minutes from previous matches only 
    - Previous_Injuries: rolling sum of past injuries only 
    Missing values are filled with medians for stability
    Both winner and loser rows are used here so that a player's full
    match history (wins and losses) informs their workload features correctly
    """
    # Days since last match per player 
    players_df["Recovery_Time"] = players_df.groupby("player_name")["date"].diff().dt.days

    # Rolling mean of minutes - shift(1) ensures current match is excluded 
    players_df["Training_Intensity"] = (
        players_df.groupby("player_name")["minutes"]
        .transform(lambda s: s.shift(1).rolling(ROLL_MINUTES_WINDOW, min_periods=1).mean())
    )

    # Rolling sum of prior injuries - shift(1) ensures current match is excluded
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

# Step 1: Load all years (train + test combined) so rolling featues are computed
# across the full timeline per player - this is important so that a player's 
# training intensity heading into a 2019 match correctly reflects their 2018 history
all_years = TRAIN_YEARS + [TEST_YEAR]
matches = load_all_matches(DATASET_FOLDER, all_years)


print("\nMatches preview:")
print(matches.head())

# Step 2: Label injuries
injury_loser = label_injuries_from_scores(matches)

# Step 3: Build player-level rows, tagging each row with its side (winner/loser)
# Both sides are kept at this stage so rolling workload features use full history
winners = make_player_rows(matches, injury_loser, side="winner")
losers = make_player_rows(matches, injury_loser, side="loser")
players = pd.concat([winners, losers], axis=0, ignore_index=True)
players = players.sort_values(["player_name", "date"]).reset_index(drop=True)

print("\nPlayer-level dataset preview:")
print(players.head())

#Quick info table (dtype, unique, null)
df_info = pd.DataFrame(players.dtypes, columns=["Dtype"])
df_info["Unique"] = players.nunique().values 
df_info["Null"] = players.isnull().sum().values
print("\nplayers info (dtype/unique/null):")
print(df_info.head(20))

# Step 4: Add workload features (uses shift(1) - no leakage) 
# Both winner and loser rows used here so each player's full match history
# informs their workload features correctly
players = add_workload_features(players)

# Step 5: EDA plots on loser rows only using Train Years (since those are what we model)
losers_eda = players[
    (players["side"] == "loser") &
    (players["date"].dt.year.isin(TRAIN_YEARS))
]

for c in ["Training_Intensity", "Recovery_Time", "Previous_Injuries", "minutes"]:
    plot_feature_distributions(losers_eda, c, target_col="Injury")

# Adding Injury Pie Chart
counts = losers_eda["Injury"].value_counts()
plt.figure(figsize=(4.2,4.3))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140,
        colors=["#1a7009", "#af0c0c"])
plt.title("Injury (0/1) Distribution - Loser Rows Only")
plt.axis("equal")
plt.show()

# Simple category counts 
plot_counts_by_category_and_label(losers_eda, "surface", hue="Injury")
plot_counts_by_category_and_label(losers_eda, "round", hue="Injury")

# Step 6: Encode categoricals with get_dummies 
category_columns = [c for c in ["surface", "round", "tourney_name"] if c in players.columns]
df = pd.get_dummies(players, columns=category_columns, dummy_na=True)

# Step 7: Filter to loser rows inly befoore assembling X and y
# This gives true match-level evaluation - one row per match
# Winners were kept above so workload features are computed correctly,
# but they are excluded from the actual prediction task here
df = df[df["side"] == "loser"].copy()

print(f"\nRows after filtering to losers only: {len(df)}")

# Step 8: Assemble features X and target y
feature_cols = [
    "minutes", "Training_Intensity", "Recovery_Time", "Previous_Injuries",
    "aces", "double_faults", "serve_points", "first_in", "first_won",
    "second_won", "serve_games", "bp_saved", "bp_faced", "rank", "rank_points", "age", "height_cm",
]

feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df["Injury"].astype(int)

print("\nTarget distribution (all years):")
print(y.value_counts())

# If no positive/ negative variation, stop
if y.nunique() < 2:
    print("\n[Warning] Target is single-class (no variation). Skipping model training")
    raise SystemExit(0)

# Step 9: Time-based Train/test split
# Train on TRAIN_YEARS, test on TEST_YEAR
# This means every test match genuinely occurred after every training match,
# making this true prospective (forward-looking injury prediction).
# This is the key different from a random split.

train_mask = df["date"].dt.year.isin(TRAIN_YEARS) # rows from training years
test_mask = df["date"].dt.year == TEST_YEAR # rows from test year

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask],  y[test_mask]

print(f"\nTime-based split (match level - loser rows only):")
print(f" Training years: {TRAIN_YEARS} -> {len(X_train)} matches")
print(f" Test year: {TEST_YEAR} -> {len(X_test)} matches")
print(f"\nTraining injury distribution: \n{y_train.value_counts()}")
print(f"\nTest injury distribution: \n{y_test.value_counts()}")

# Safety check - need both classes in train and test
if y_train.nunique() < 2 or y_test.nunique() < 2:
    print("\n[Warning] One of the splits has only a single class."
          "Check your yearly CSV files are loading correctly.")
    raise SystemExit(0)

# Step 10: Define and train models: ExtraTrees, NuSVC (with scaling), LGBM 
class_frac = y_train.value_counts(normalize=True)
minority_frac = float(class_frac.min()) if len(class_frac) > 1 else 0.1
feasible_nu = max(0.01, min(0.49, minority_frac - 1e-3)) # v must be feasible re. class balance

models = {
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "NuSVC": make_pipeline(
        StandardScaler(),
        NuSVC(probability=True, nu=feasible_nu, kernel="rbf")
    ),
    "LGBM": LGBMClassifier(
        random_state=RANDOM_STATE
        # class_weight="balanced" # uncomment if injury class is very rare
    ),
}

# Train, evaluate, and collect probabilities
aucs = {}
probas = {}

print("\n== Model Performance (Trained on past, tested on future) ==")
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Basic metrics 
    accuracy =  accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0) # TPR (sensitivity)
    f1 = f1_score(y_test, predictions, zero_division=0)

    # Confusion matrix and negative-side metrics
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    # Specificity (TNR) and False Negative Rate (FNR)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f} (TPR)")
    print(f"Specific: {tnr:.4f} (TNR)")
    print(f"FNR: {fnr:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Confusion matrix counts -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")
    print(f"Total test matches: {tn+fp+fn+tp}")
    print("Classification report:")
    print(classification_report(y_test, predictions, digits=4, zero_division=0))

    # Confusion matrixes
    plt.figure(figsize=(4.2,3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}\n(Train: {TRAIN_YEARS} | Test: {TEST_YEAR})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    
    # Probabilities for ROC (if no predict_proba, scale decision_function to [0,1])
    if hasattr(model, "predict_proba"):
        pos_proba = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        pos_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    auc = roc_auc_score(y_test, pos_proba)
    aucs[model_name] = float(auc)
    probas[model_name] = pos_proba

# ROC Curve: TPR vs FPR 
plt.figure(figsize=(7,5.5))
for model_name, pos_proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, pos_proba)
    auc = roc_auc_score(y_test, pos_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.2f})")
    
# Plot the diagonal line
plt.plot([0, 1], [0, 1], linestyle='--', color='black')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title(f"ROC Curves - Trained on {TRAIN_YEARS}, Tested on {TEST_YEAR}")
plt.legend()
plt.tight_layout()
plt.show()

# Precision Recall Curve (better for rare positives like injuries)
plt.figure(figsize=(7,5.5))
for model_name, pos_proba in probas.items():
   precision_vals, recall_vals, _ = precision_recall_curve(y_test, pos_proba)
   ap = average_precision_score(y_test, pos_proba)
   plt.plot(recall_vals, precision_vals, label=f"{model_name} (AP={ap:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curves - Trained on {TRAIN_YEARS}, Tested on {TEST_YEAR}")
plt.legend()
plt.tight_layout()
plt.show()



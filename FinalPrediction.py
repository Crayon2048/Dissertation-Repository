# FinalPrediction.py
# - Loads ATP matches (Jeff Sackmann's dataset) from a CSV file
# - Builds a player-level dataset (winner + loser rows)
# - Labels "injury" when score has RET/W/O (loser only)
# - Plots distributions using histograms, KDE and boxplots
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
from sklearn.metrics import roc_curve, roc_auc_score

# Configuration / constants (easy to edit)
CSV_FILEPATH = 'C:/Users/kleon/OneDrive/Documents/Dissertation/Dissertation-Repository/Datasets/atp_matches_2019.csv'
RANDOM_STATE = 42
ROLL_MINUTES_WINDOW = 5 # window for rolling mean of minutes played
ROLL_INJURIES_WINDOW = 10 # window for rolling sum of previous injuries
TEST_SIZE = 0.2 # 80/20 train/test split

sns.set(style="whitegrid")

# Libraries and Python version
library = {
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

for nome, library in sorted(library.items()):
    print(f"{nome:<20} | {library.__version__:>10}")

# Helper functions

def load_matches(path):
    """
    Load ATP matches from a CSV file and converts tourney_date to a date time (YYYYMMDD) -> Timestamp).
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
    return score_str.str.contain(r"RET|W/O", case=False, na=False).astype(int)

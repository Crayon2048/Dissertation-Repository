# FinalPrediction.py
# - Loads ATP matches (Jeff Sackmann's dataset) from a CSV file
# - Builds a player-level dataset (winner + loser rows)
# - Labels "injury" when score has RET/W/O (loser only)
# - Plots distributions using histograms, KDE and boxplots
# - Trains three models this time (ExtraTrees, LightGBM, NuSVC) and evaluates them
# - Will need to make confustion matrices, ROC (TPR VS FPR) curves,
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

# Python Version    
print()
print(f"Python Version: {python_version()}")

# Importing df 
df = pd.read_csv('C:/Users/kleon/OneDrive/Documents/Dissertation/Dissertation-Repository/Datasets/expanded_injury_workload.csv')

# Preview
print("\nDataset preview:")
print(df.head())

# Dataset cleaning
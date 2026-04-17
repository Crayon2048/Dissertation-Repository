# How can AI help predict possible injuries for tennis athletes to improve performance? 

**Dissertation Project - Kleone Gayya | Student Number: 19293737**

This project investigates how artificial intelligence can be used to predict injury risk in professional tennis players, with the broader goal of informing performance improvement decisions such as training load management, rest scheduling, and match preparation. It uses historical match data from [Jeff Sackmann's tennis dataset](https://github.com/JeffSackmann/tennis_atp)

---

## Overview 

Tennis is a physically demanding individual sport requiring repetitive high intensity movements, explosive acceleration, rapid directional changes, and prolonged match durations. These demands place players at significant risk of musculoskeletal injuries, particularly overuse injuries affecting the shoulder, elbow, knee, and lower back. At the elite level, even minor injuries can significantly impact competitive performance and long-term athlete development. 

Injuries in professional tennis are often only identified after a player retires mid match or withdraws before one, at which point the damage to both performance and career continuity has already occured. Traditional injury prevention approaches often rely on retrospective analysis or subjective judgement, which may fail to capture early indicators of increased risk. 

This project builds a machine learning pipeline that predicts whether an ATP tennis player is at risk of retiring or withdrawing from a match (RET/W/O), framed as a binary classification problem. By identifying patterns in match statisitics, player workload, and historical injury history, the system aims to provide actionable risk signals that could support performance and welfare decisions. AI-based predicitive models offer the potential to complement exisiting methods by analysing large volumes of historical data to detect subtle trends associated with injury occurence. 

The key methodological decision is a **time-based train/test split:** models are trained on 2017-2018/ 2021-2022 data and tested on 2019/2023 as genuinely unseen future matches, making this a true prospective injury prediction system rather than a retrospective classification exercise. 

Note: This project does not aim to produce a clinically validated system. It demonstrates the feasibility and potential of AI-driven injury prediction in tennis, and predictions should be interpreted as probabilistic risk indicators rather than definitive diagnoses. 

## Research Context

Much of the existing AI-driven injury prediciton research focuses on team sports such as football and basketball, where GPS tracking data, physiological measurements, and detailed medical records are more readily available. Injury prediction models tailored specifically to the unique biomechanical and competitive demands of tennis remain limited. 

This project addresses that gap by adapting machine learning techniques to a tennis specific dataset, priortising sport relevant features such as: match duration surface type, recovery time, and workload accumulation. That reflect the physical demands unique to professional tennis. 

# Dataset

- **Source:** [Jeff Sackmann's ATP matches repository] (https://github.com/JeffSackmann/tennis_atp) 
- **Files used:** atp_matches_2017.csv, atp_matches_2018.csv, atp_matches_2019, atp_matches_2021.csv, atp_matches_2022.csv, atp_matches_2023.csv. 
- **Injury label:** A match is flagged as an injury event if the score column contains RET (retirement) or W/O (walkover), affecting the losing player only. 
- All datasets are used in accordance with their licesning terms (Creative Commons Attribution 4.0). No proprietary or sensitive medical data is included. 

## Project Structure 

```
Dissertation Final Code/
  FinalPrediction.py.txt
Dissertation Repository/
  Datasets/
    atp_matches_2017.csv 
    atp_matches_2018.csv
    atp_matches_2019.csv
    atp_matches_2021.csv
    atp_matches_2022.csv 
    atp_matches_2023.csv 
  Screenshots for Testing/
    After Modifications
    Before Modificatons
  FinalPrediction.py
  Prediction2.py
  README.md
```

## How It Works

### 1. Data Loading
Multiple yearly CSV files are loaded and combined into a single DataFrame. The `tourney_date` column is converted from integer format (YYYYMMDD) to a proper datetime to enable time based operations. 

### 2. Injury Labelling
The score column is scanned for `RET` or `W/O` strings. Any match containing these is flagged as an injury event (Injury = 1). Only the losing players recieves this label since retirements are always recorded as losses. 

### 3. Player Level Rows
Each match is expanded into two rows, one for the winner and one for the loser, so that each row captures one player's individual match statistics (aces, double faults, rank, and age). A `side` column tracks whether each row came from the winner or loser perspective. 

### 4. Workload Features
Three features are engineered per player using only their **past** match history. `shift(1)` is applied throughout to prevent data leakage - the current match is never included in its own feature calculation. Workload accumulation across training and competition has been identified in the literature as a significant contributor to injury risk, particulalry in elite athletes who compete year round:

| Feature  | Description |
| ------------- | ------------- |
| `Recovery_Time`  |Days since the player's last match |
| `Training_Intensity`  | Rolling mean of minutes played across the last 5 matches  |
| `Previous_Injuries`  | Rolling count of retirements across the last 10 seasons |

### 5. Exploratory Data Analysis
Distribution plots (histograms, KDE, boxplot) and category count charts are generated to understand injury patterns across surfaces and rounds. EDA is restricted to **2021-2022 training data only** so the 2023 test set remains completely unseen at this stage.

### 6. Match Level Filtering
Before modelling, winner rows are dropped so that each remaining row corresponds to exactly one match. This gives true match level evaluation where confusion matrix numbers directly corresponds to real match counts rather than inflated player match counts. 

### 7. Time based Train/Test Split
- **Training set:** 2021-2022 loser rows
- **Test set:** 2023 loser rows (genuinely unseen future matches)

This mirrors how the system would work in practice - trained on historical data, evaluated on future matches it has never encountered. 

### 8. Models
Three classifiers are trained and compared, reflecting the trade-off between accuracy and interpretability identified in the literature:

| Model  | Notes |
| ------------- | ------------- |
| **ExtraTrees**  | 200 randomised decision trees with `class_weight=balanced` to handle injury class imbalance |
| **LightGBM**  | Sequential gradient boosting, generally fast and strong on tabular data |
| **NuSVC** | Support vector classifier with RBF kernel, scaled with `StandardScaler`. `nu` is calculated automatically from class balance to prevent errors |

### 9. Evaluation
Each model is evaluated on the 2023 test set with the following metrics:

- Accuracy
- Precision
- Recall (TPR / Sensitivity)
- Specificity (TNR)
- False Negative Rate (FNR)
- F1-Score
- Confusion Matrix
- ROC Curve with AUC Score
- Precision-Recall Curve

## Results Summary

The test set comprised 2,986 loser rows from 2023, of which 96 carried an injury label - a class imbalance ratio of approximately 97:3.


| Model  | TP | FP | FN | Recall | Precision | AUC-ROC | Avg Precision |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ExtraTreesClassifier  | 41 | 2 | 55 | 42.7% | 95.3% | 0.90 | 0.72 |
| LightGBM  | 46 | 4 | 50 | 47.9% | 92.0% | 0.90 | 0.70 |
| NuSVC | 48 | 29 | 48 | 50.0% | 62.3% | 0.87 | 0.58 |

ExtraTreesClassifier and LightGBM both acheived an AUC-ROC of 0.90. LightGBM offered the best overall precision-recall balance at the default threshold (47.9% recall, 92.0% precision, only 4 false positives), making it the most operationally practical model for deploymenet. NuSVC achieved the highest raw recall (50%) but generated significantly more false positives (29).

---

## Key Design Decisions 

**Why loser rows only for modelling?**
Winners can never have Injury = 1 in this dataset, so including them would artifically inflate the True Negative count and make evaluation misleading. Filtering to losers only gives one meaningful row per match.

**Why shift(1) on workload features?**
Without `shift(1)`, the current match's minutes would be included in its own Training_Intensity calculation - information that wouldn't exist at prediction time. This would consistute data leakage and artifically inflate model performance. 

**Why a time-based split instead of random?**
A random split could place a January 2023 match in the training set and a December 2022 match in the test set, which makes no sense for prediction. A time-based split ensures every test match genuinley ocured after every training match. 

**Why restrict EDA to training years?**
Plotting injury patterns from 2023 before model evaluation technically gives prior knowledge of the test set. Restricting EDA to 2021-2022 keeps the methodology clean and fully prospective. 

**Why these three models?**
The literature identifies a trade-off between accuracy and interpretability in injury prediction models. Neural networks often achieve higher performance but lack transparency, reducing practical usefulness for coaches. Tree-based models like ExtraTrees and LightGBM offer strong accuracy with greater interpretability. NuSVC provides a contrasting boundary based approach for comparison. 

---

## How Predictions Could Improve Performance

The predictions generated by this system are intended to support real-world performance and welfare decisions, not replace human judgement. A high injury risk score for a player heading into a match could prompt several interventions:

- **Coaching staff** could reduce training intensity or volume in the days before a tournament
- **Medical teams** could schedule additional physiotherapy or screening for flagged players
- **Tournament schedulers** could consider rest days or adjusted match scheduling for at risk players
- **Performance analysts** could monitor workload trends across a season to identify players accumulating dangerous levels of fatigue

Outputs are framed as supplementary decision support information rather than prescriptive directives. This approach aligns with professional guidance from the BCS Code of Conduct and ACM Code of Ethics, which emphasise accountablity, fairness, and the responsible deployment of AI systems within real-world contexts.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
```
Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
```

---

## Usage

1. Download the ATP match CSV files from [Jeff Sackmann's ATP matches repository](https://github.com/JeffSackmann/tennis_atp) and place them in your Datasets folder
2. Update the `DATASET_FOLDER` path in `FinalPrediction.py` to point to your local folder
3. Run the script:
```bash
python FinalPrediction.py
```

---

## Limitations

- **Injury label is a proxy** - RET/W/O in the score column does not always indicate a physical injury. A player may retire due to illness, personal reasons, or tactical decisions in rare cases
- **No within match physiological data** - all match statistics are recorded at match level, there is no measure of heart rate, movement distance, or perceived exertion
- **Class imbalance** - injuries are rare events (approximately 3% of matches), which makes them inherently difficult to predict and means accuracy alone is a misleading metric
- **Single sport and tour** - the model is trained only on ATP men's tour level matches and may not generalise to WTA, ITF, Challenger events, or other tennis tours. 
- **Generalisability** - models trained on elite male ATP data may not generalise to female athletes, junior players, or players at different competitive levels 
- **No medical validation** - predictions are probabilistic indicators of elevated risk, not clinically validated diagnoses
- **Single held out year** - evaluation is based on one test year (2023), so no confidence intervals are reported around AUC or Average Precision Values 

--- 

## Ethical Considerations

This project adheres to professional codes of conduct outlined by the British Computer Society (BCS) and the Association for Computing Machinery (ACM), as well as the EU Ethics Guidelines for Trustworthy AI. Key ethical considerations include:

- Predictions are framed as decision support tools rather than automated judgements 
- Model limitations are explicitly communicated to prevent misinterpretation 
- All datasets are used in accordance with their licensing terms (CC BY 4.0)
- No proprietary or sensitive personal data is included 
- The system is not intended to influence selection decisions or disadvantage individual athletes
- From an EDI perspective, the dataset covers only ATP tour male players, the system should not be applied to WTA, junior, or sub-elite populations without retraining
- Any deployment in a professional sporting context would require formal legal review, including a Data Protection Impact Assessment under UK GDPR if physiological data were incorporated

---

## Author

**Kleone Gayya** | Student Number: 19293737

Dissertation project - *How can AI help predict possible injuries for tennis athletes to improve performance?*

Version management: [GitHub Repository](https://github.com/Crayon2048/Dissertation-Repository)
How can AI help predict possible injuries for tennis athletes to improve performance? 

Dissertation Project - Kleone Gayya | Student Number: 19293737

This project investigates how artificial intelligence can be used to predict injury risk in professional tennis players, with the broader goal of informing performance improvement decisions such as training load management, rest scheduling, and match preparation. It uses historica match data from Jeff Sackmann's tennis dataset

Overview 

Tennis is a physically demanding individual sport requiring repetitive high intensity movements, explosive acceleration, rapid directional changes, and prolonged match durations. These demands place players at significant risk of musculoskeletal injuries, particularly overuse injuries affecting the shoulder, elbow, knee, and lower back. At the elite level, even minor injuries can significantly impact competitive performance and long-term athlete development. 

Injuries in professional tennis are often only identified after a player retires mid match or withdraws before one, at which point the damage to both performance and career continuity has already occured. Traditional injury prevention approaches often rely on retrospective analysis or subjective judgement, which may fail to capture early indicators of increased risk. 

This project builds a machine learning pipeline that predicts whether an ATP tennis player is at risk of retiring or withdrawing from a match (RET/W/O), framed as a binary classification problem. By identifying patterns in match statisitics, player workload, and historical injury history, the system aims to provide actionable risk signals that could support performance and welfare decisions. AI-based predicitive models offer the potential to complement exisiting methods by analysing large volumes of historical data to detect subtle trends associated with injury occurence. 

The key methodological decision is a time-based train/test split: models are trained on 2017-2018/ 2021-2022 data and tested on 2019/2023 as genuinely unseen future matches, making this a true prospective injury prediction system rather than a retrospective classification exercise. 

Note: This project does not aim to produce a clinically validated system. It demonstrates the feasibility and potential of AI-driven injury prediction in tennis, and predictions should be interpreted as probabilistic risk indicators rather than definitive diagnoses. 

Research Context

Much of the existing AI-driven injury prediciton research focuses on team sports such as football and basketball, where GPS tracking data, physiological measurements, and detailed medical records are more readily available. Injury prediction models tailored specifically to the unique biomechanical and competitive demands of tennis remain limited. 

This project addresses that gap by adapting machine learning techniques to a tennis specific dataset, priortising sport relevant features such as: match durationm surface type, recovery time, and workload accumulation. That reflect the physical demands unique to professional tennis. 

Dataset

Source: https://github.com/JeffSackmann/tennis_atp 
Files used: atp_matches_2017.csv, atp_matches_2018.csv, atp_matches_2019, atp_matches_2021.csv, atp_matches_2022.csv, atp_matches_2023.csv. 
Injury label: A match is flagged as an injury event if the score column contains RET (retirement) or W/O (walkover), affecting the losing player only. 
All datasets are used in accordance with their licesning terms (Creative Commons Attribution 4.0). No proprietary or sensitive medical data is included. 

Project Structure 

Dissertation Final Code/
  # FinalPrediction.py
Dissertation Repository/
  Datasets/
    atp_matches_2017.csv 
    atp_matches_2018.csv
    atp_matches_2019.csv
    atp_matches_2021.csv
    atp_matches_2022.csv 
    atp_matches_2023.csv. 
  Screenshots for Testing/
    After Modifications
    Before Modificatons
  FinalPrediction.py
  Prediction2.py
  README.md

How It Works

1. Data Loading

2. Injury Labelling

3. Player Level Rows

4. Workload Features

5. Exploratory Data Analysis

6. Match Level Filtering

7. Time based Train/Test Split

8. Models

9. Evaluation


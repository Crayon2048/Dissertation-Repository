How can AI help predict possible injuries for tennis athletes to improve performance? 

Dissertation Project - Kleone Gayya | Student Number: 19293737

This project investigates how artificial intelligence can be used to predict injury risk in professional tennis players, with the broader goal of informing performance improvement decisions such as training load management, rest scheduling, and match preparation. It uses historica match data from Jeff Sackmann's tennis dataset

Overview 

Tennis is a physically demanding individual sport requiring repetitive high intensity movements, explosive acceleration, rapid directional changes, and prolonged match durations. These demands place players at significant risk of musculoskeletal injuries, particularly overuse injuries affecting the shoulder, elbow, knee, and lower back. At the elite level, even minor injuries can significantly impact competitive performance and long-term athlete development. 

Injuries in professional tennis are often only identified after a player retires mid match or withdraws before one, at which point the damage to both performance and career continuity has already occured. Traditional injury prevention approaches often rely on retrospective analysis or subjective judgement, which may fail to capture early indicators of increased risk. 

This project builds a machine learning pipeline that predicts whether an ATP tennis player is at risk of retiring or withdrawing from a match (RET/W/O), framed as a binary classification problem. By identifying patterns in match statisitics, player workload, and historical injury history, the system aims to provide actionable risk signals that could support performance and welfare decisions. AI-based predicitive models offer the potential to complement exisiting methods by analysing large volumes of historical data to detect subtle trends associated with injury occurence. 

The key methodological decision is a time-based train/test split: models are trained on 2017-2018/ 2021-2022 data and tested on 2019/2023 as genuinely unseen future matches, making this a true prospective injury prediction system rather than a retrospective classification exercise. 

Note: This project does not aim to produce a clinically validated system. It demonstrates the feasibility and potential of AI-driven injury prediction in tennis, and predictions should be interpreted as probabilistic risk indicators rather than definitive diagnoses. 

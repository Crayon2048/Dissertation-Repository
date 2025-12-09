import pandas as pd

# 1. Load original injury data and rallies

injury_data_path = "C:/Users/kleon/OneDrive/Documents/Dissertation/Dissertation-Repository/injury_data.csv"
rallies_path = "C:/Users/kleon/OneDrive/Documents/Dissertation/Dissertation-Repository/rallies.csv"

injury_data = pd.read_csv(injury_data_path)
rallies = pd.read_csv(rallies_path)

# 2. Create a version with only Djokovic and Nadal 

injury_data_players = injury_data.sample(2, random_state=42).copy()
injury_data_players["server"] = ["Djokovic", "Nadal"]

# 3. Replace attributes with AO 2019 accurate stats 
player_stats_ao2019 = {
    "Djokovic": {
        "Player_Age": 31,
        "Player_Height": 188.0,
        "Player_Weight": 77.0,
        "Previous_Injuries": 1,
        "Training_Intensity": 0.85,
        "Recovery_Time": 2,
    },
    "Nadal": {
        "Player_Age": 32,
        "Player_Height": 185.0,
        "Player_Weight": 85.0,
        "Previous_Injuries": 2,
        "Training_Intensity": 0.75,
        "Recovery_Time": 3,
    },
}

#Apply player stats
for player, stats in player_stats_ao2019.items():
    mask = injury_data_players["server"] == player
    for col, val in stats.items():
        injury_data_players.loc[mask,col] = val

#Add tournament info
injury_data_players["Tournament"] = "Australian Open 2019"

# 4. Save updated injury data 

injury_data_players_path = "C:/Users/kleon/OneDrive/Documents/Dissertation/Dissertation-Repository/injury_data_players.csv"
injury_data_players.to_csv(injury_data_players_path, index=False)
print("Updated and saved injury_data_players.csv")
print(injury_data_players, "\n")

# 5. Clean rally data and compute A0 2019 workload

rallies_clean = rallies.replace("_undefined_", pd.NA)
mask_ao2019 = rallies_clean.apply(
    lambda r: ("australian" in str(r.values).lower()) or ("2019" in str(r.values)), axis=1 
)
rallies_ao2019 = rallies_clean[mask_ao2019].copy()

if len(rallies_ao2019) == 0:
    print("No AO2019 specific rallies found - using all rallies instead")
    rallies_ao2019 = rallies_clean

#  6. Create rally level features 

rallies_ao2019["is_first_serve"] = (rallies_ao2019["serve"] == "first").astype(int)
rallies_ao2019["is_error"] = (rallies_ao2019["reason"] == "out").astype(int)


# 7. Expanded merge (no groupby!)

expanded_merged = rallies_ao2019.merge(injury_data_players, on="server", how="left")

# 8. Save the merged dataset 

expanded_path = "C:/Users/kleon/OneDrive/Documents/Dissertation/Dissertation-Repository/expanded_injury_workload.csv"
expanded_merged.to_csv(expanded_path, index=False)
print("Expanded dataset saved:", expanded_path)
print("Rows", len(expanded_merged))
print(expanded_merged.head())
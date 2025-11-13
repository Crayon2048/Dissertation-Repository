import pandas as pd

# 1. Load original injury data and rallies

injury_data = pd.read_csv("C:/Users/kleon/OneDrive/Documents/Dissertation/Dissertation-Repository/injury_data.csv")
rallies = pd.read_csv("C:/Users/kleon/OneDrive/Documents/Dissertation/Dissertation-Repository/rallies.csv")

# 2. Create a version with only Djokovic and Nadal 

injury_data_players = injury_data.sample(2, random_state=42).copy()
injury_data_players["server"] = ["Djokovic", "Nadal"]

# Save this reduced version 

injury_data_players.to_csv("injury_data_players.csv", index=False)
print("Created 'injury_data_players.csv'")
print(injury_data_players, "\n")

# 3 Clean rally data

rallies = rallies.replace("_undefined_", pd.NA)

#  4. Compute player-level workload metrics 

rally_features = (
    rallies.groupby("server")
    .agg(
        avg_strokes_per_rally=("strokes", "mean"),
        avg_rally_time=("totaltime", "mean"),
        first_serve_ratio=("serve", lambda x: (x == "first").mean()),
        error_rate=("reason", lambda x: (x == "out").mean()),
        total_rallies=("rallyid", "count"),
    )
    .reset_index()
)

print("Rally-level workload features:")
print(rally_features, "\n")

# 5. Merge injury and rally data

merged_data = injury_data_players.merge(rally_features, on="server", how="left")

# 6. Save the merged dataset 

merged_data.to_csv("merged_injury_workload.csv", index=False)

print("Merged dataset saved as 'merged_injury_workload.csv'")
print(merged_data)


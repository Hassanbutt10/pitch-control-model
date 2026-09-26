"""
Pitch Control Model - Cycle 5, Day 3 project
Grid-based static pitch control (nearest-player assignment, the Voronoi
approach) using StatsBomb 360 freeze-frame data.

Limitation: 360 data only marks players as teammate/opponent/actor with a
location, no player ID across frames. There is no way to track a player
between frames, so velocity/direction cannot be estimated from this open
dataset. This model uses static position only, same limitation the packing
model had with x-axis-only tracking.

Reuses: frames_final.csv, events_final.csv (copied from packing-rate-calculator)
"""

import pandas as pd
import numpy as np
import ast

frames = pd.read_csv("frames_final.csv", low_memory=False)
events = pd.read_csv("events_final.csv", low_memory=False)

def parse_loc(val):
    if pd.isna(val):
        return None, None
    try:
        coords = ast.literal_eval(val)
        return coords[0], coords[1]
    except (ValueError, SyntaxError, IndexError):
        return None, None

frames[["x", "y"]] = frames["location"].apply(lambda v: pd.Series(parse_loc(v)))
frames = frames.dropna(subset=["x", "y"])

# event context: which team was on the ball for each frame id
event_info = events.set_index("id")[["team", "minute", "type"]]
event_info = event_info[~event_info.index.duplicated(keep="first")]

teams = events["team"].dropna().unique()
if len(teams) != 2:
    raise ValueError(f"Expected 2 teams, found: {teams}")
team_a, team_b = teams[0], teams[1]

# ---------- grid setup ----------
PITCH_X, PITCH_Y = 120, 80
STEP = 2
grid_x, grid_y = np.meshgrid(np.arange(0, PITCH_X, STEP), np.arange(0, PITCH_Y, STEP))
grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

# ---------- per-frame control calculation ----------
records = []
for frame_id, group in frames.groupby("id"):
    if frame_id not in event_info.index:
        continue
    actor_team = event_info.loc[frame_id, "team"]
    minute = event_info.loc[frame_id, "minute"]
    opponent_team = team_b if actor_team == team_a else team_a

    own_players = group[group["teammate"] == True][["x", "y"]].values
    opp_players = group[group["teammate"] == False][["x", "y"]].values

    if len(own_players) == 0 or len(opp_players) == 0:
        continue

    dist_own = np.min(
        np.linalg.norm(grid_points[:, None, :] - own_players[None, :, :], axis=2), axis=1
    )
    dist_opp = np.min(
        np.linalg.norm(grid_points[:, None, :] - opp_players[None, :, :], axis=2), axis=1
    )

    own_control_pct = np.mean(dist_own < dist_opp) * 100

    records.append({"frame_id": frame_id, "minute": minute, "team": actor_team, "control_pct": own_control_pct})
    records.append({"frame_id": frame_id, "minute": minute, "team": opponent_team, "control_pct": 100 - own_control_pct})

control_df = pd.DataFrame(records)
control_df.to_csv("frame_control.csv", index=False)

# ---------- whole-match average ----------
avg_control = control_df.groupby("team")["control_pct"].mean().sort_values(ascending=False)
print("\nAverage pitch control across the match:")
print(avg_control)

# ---------- pick a snapshot to visualize: frame with the most players visible ----------
frame_counts = frames.groupby("id").size()
hero_id = frame_counts.idxmax()
hero_event = event_info.loc[hero_id]
if isinstance(hero_event, pd.DataFrame):
    hero_event = hero_event.iloc[0]
hero_frame = frames[frames["id"] == hero_id]

hero_own = hero_frame[hero_frame["teammate"] == True][["x", "y"]].values
hero_opp = hero_frame[hero_frame["teammate"] == False][["x", "y"]].values

dist_own = np.min(np.linalg.norm(grid_points[:, None, :] - hero_own[None, :, :], axis=2), axis=1)
dist_opp = np.min(np.linalg.norm(grid_points[:, None, :] - hero_opp[None, :, :], axis=2), axis=1)
control_grid = (dist_own < dist_opp).reshape(grid_y.shape)

# ---------- visualization ----------
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 7.3))
fig.patch.set_facecolor("#0d1b2a")
ax.set_facecolor("#0d1b2a")

ax.contourf(grid_x, grid_y, control_grid, levels=1, colors=["#f4a300", "#4a90d9"], alpha=0.55)
ax.scatter(hero_own[:, 0], hero_own[:, 1], color="#4a90d9", s=90, edgecolors="white", label="Attacking team (on ball)", zorder=5)
ax.scatter(hero_opp[:, 0], hero_opp[:, 1], color="#f4a300", s=90, edgecolors="white", label="Opposing team", zorder=5)

ax.set_xlim(0, PITCH_X)
ax.set_ylim(0, PITCH_Y)
ax.set_title(f"Pitch Control Snapshot: {hero_event['type']} in Minute {hero_event['minute']}",
             color="white", fontsize=13)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("#3a4a5a")
ax.legend(facecolor="#0d1b2a", labelcolor="white", edgecolor="#3a4a5a", loc="upper right")

plt.tight_layout()
plt.savefig("pitch_control_snapshot.png", dpi=150, facecolor=fig.get_facecolor())
print("\nSaved chart: pitch_control_snapshot.png")

# ---------- second chart: average control bar ----------
fig2, ax2 = plt.subplots(figsize=(8, 5.5))
fig2.patch.set_facecolor("#0d1b2a")
ax2.set_facecolor("#0d1b2a")
colors = ["#4a90d9", "#f4a300"]
ax2.bar(avg_control.index, avg_control.values, color=colors[:len(avg_control)])
ax2.set_ylabel("Average pitch control (%)", color="white")
ax2.set_title("Average Pitch Control Across the Match", color="white", fontsize=13)
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_color("#3a4a5a")

plt.tight_layout()
plt.savefig("avg_pitch_control.png", dpi=150, facecolor=fig2.get_facecolor())
print("Saved chart: avg_pitch_control.png")
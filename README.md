# Pitch Control Model: Argentina vs France, WC 2022 Final

## Why
Possession stats only count who has the ball. They say nothing about who controls the space around it. This project builds a pitch control model to measure exactly how much of the pitch each team owned at any given moment, using the geometry of where every player stood.

## What it does
- Uses StatsBomb 360 freeze-frame data to get every player's position at each event
- Builds a grid across the pitch and assigns each point to whichever player (teammate or opponent) is nearest to it, the core idea behind Voronoi-based pitch control
- Calculates each team's control percentage frame by frame, then averages it across the whole match
- Renders a snapshot of one frame showing the actual control boundary between the two teams

## Key findings
| Team | Average pitch control |
|---|---|
| France | 52.1% |
| Argentina | 47.9% |

- France controlled more of the pitch on average across the match, despite Argentina winning the game, showing that space control and match outcome are not the same thing
- The snapshot frame (minute 92, a Pressure event) shows the mosaic pattern this method produces: tightly packed players create jagged, contested boundaries, while isolated players claim large uncontested zones

## Limitation
StatsBomb's open 360 data does not assign player IDs across frames, only teammate/opponent/actor flags with a location. This makes it impossible to track an individual player between frames, so velocity and direction cannot be estimated. This model uses static position only, not the velocity-adjusted version used in professional tracking-data systems.

## Visual
`pitch_control_snapshot.png`: control boundary at a single frame. `avg_pitch_control.png`: average control percentage across the match.

## Tools
Python, pandas, numpy, matplotlib

## Data source
StatsBomb open data (via `statsbombpy`), World Cup 2022 Final, 360 freeze-frame data

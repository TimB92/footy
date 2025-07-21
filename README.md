# EURO 2024 Match Predictor
This projects does an attempt at predicting football matches for EURO 2024, based on historical international matches. The lightweight model estimates team skill over time, incorporates tournament-level effects, and accounts for recent team form—replacing traditional notions like home advantage, which are less relevant in international competitions. 

This is a hobby project and the model is extremely basic, not taking into account things like players, rankings and other factors that influence a match result. The goal was to create a model using as little data as possible, while having fun with Bayesian statistics. Ironically, it beat all of my colleagues' advanced NN's in the scorito group of the company I work at, showcasing the power of simplicity. 

# Features
Bayesian Inference with PyMC: Uses probabilistic modeling to estimate latent team skills.

Skill effects: Models team strength over multi-year buckets.

Tournament Effects: Captures difficulty variations across competitions (e.g. UEFA Euro vs. friendlies).

Recent Form Impact: Incorporates short-term team performance (e.g. past 5-match record).

Poisson Goal Modeling: Predicts goal counts and match outcomes.


# Usage
To run from the command line, use:
```
make init
```
to initialize the repo (requires UV-package manager). Subsequently, run:
```
make run/train
```
to train the model.

You can also edit the make commands with your own input and output paths in the `Makefile`.

# Data

This assumes you have data in the form of historical match results, which you can get from anywhere on the internet. The following columns are required:

| Column       | Type     | Description                                     |
| ------------ | -------- | ----------------------------------------------- |
| `date`       | string   | Match date in `YYYY-MM-DD` format               |
| `home_team`  | string   | Name of the home team                           |
| `away_team`  | string   | Name of the away team                           |
| `home_score` | int      | Goals scored by the home team                   |
| `away_score` | int      | Goals scored by the away team                   |
| `tournament` | string   | Name of the tournament (e.g., "FIFA World Cup") |

Regarding the test set, this assumes you want to predict one round of matches. Because it includes a form factor, which is based on the last couple of matches, it will not work when the form is based on matches that have not yet been played. So if you want to predict multiple rounds, or the entire tournament in one go, you need to calculate the form after each match prediction, and use that to calculate the form factor.
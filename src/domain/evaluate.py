import numpy as np
import pandas as pd
from scipy.stats import poisson

from domain.preprocess import TeamData

def predict(trace: dict, test_data: TeamData):
    skills = trace.posterior['skill'].mean(dim=["chain", "draw"])
    baseline = trace.posterior['baseline'].mean().item()
    tournament_eff = trace.posterior['tournament_eff'].mean(dim=["chain", "draw"])
    form_coeff = trace.posterior['form'].mean(dim=["chain", "draw"])  

    predictions = []

    for match_num in range(len(test_data.home.teams)):
        home = test_data.home.teams[match_num]
        away = test_data.away.teams[match_num]
        year = test_data.year[match_num]
        tournament = test_data.tournament[match_num]
        home_form = test_data.home.form[match_num]
        away_form = test_data.away.form[match_num]

        skill_h = skills[year, home].item()
        skill_a = skills[year, away].item()
        tour_eff = tournament_eff[tournament].item()

        form_effect = form_coeff[0].item() * home_form - form_coeff[1].item() * away_form

        lambda_h = np.exp(baseline + form_effect + skill_h - skill_a + tour_eff)
        lambda_a = np.exp(baseline - form_effect + skill_a - skill_h + tour_eff)

        pred_home = poisson.rvs(mu=lambda_h, size=5000)
        pred_away = poisson.rvs(mu=lambda_a, size=5000)

        predictions.append({
            "mean_home_goal": np.mean(pred_home),
            "mean_away_goal": np.mean(pred_away),
            "mode_home_goal": np.argmax(np.bincount(pred_home)),
            "mode_away_goal": np.argmax(np.bincount(pred_away)),
            "pred_result": np.sign(np.mean(pred_home) - np.mean(pred_away))
        })

    return pd.DataFrame(predictions)

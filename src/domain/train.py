from dataclasses import dataclass

import pymc as pm

from domain.preprocess import TeamData, Mappings


@dataclass
class ModelResult:
    model: pm.Model
    trace: dict


def train(train_data: TeamData, mappings: Mappings):
    n_tournaments = len(mappings.tournament.keys())
    n_years = len(mappings.year.keys())
    n_teams = len(mappings.team.keys())

    with pm.Model() as model:
        baseline = pm.Normal("baseline", mu=0, sigma=1)

        form_effect = pm.Normal("form", mu=0.2, sigma=0.5, shape=2)

        tournament_effect = pm.Normal(
            "tournament_eff", mu=0, sigma=0.5, shape=n_tournaments
        )
        skill = pm.Normal("skill", mu=0, sigma=1, shape=(n_years, n_teams))

        form_effect = (
            form_effect[0] * train_data.home.form
            - form_effect[1] * train_data.away.form
        )

        lambda_home = pm.Deterministic(
            "lambda_home",
            pm.math.exp(
                baseline
                + form_effect
                + skill[train_data.year, train_data.home.teams]
                - skill[train_data.year, train_data.away.teams]
                + tournament_effect[train_data.tournament]
            ),
        )
        lambda_away = pm.Deterministic(
            "lambda_away",
            pm.math.exp(
                baseline
                - form_effect
                + skill[train_data.year, train_data.away.teams]
                - skill[train_data.year, train_data.home.teams]
                + tournament_effect[train_data.tournament]
            ),
        )

        home_obs = pm.Poisson(
            "home_goals", mu=lambda_home, observed=train_data.home.goals
        )
        away_obs = pm.Poisson(
            "away_goals", mu=lambda_away, observed=train_data.away.goals
        )

        trace = pm.sample(
            1000, tune=1000, chains=4, target_accept=0.9, return_inferencedata=True
        )

    return ModelResult(model, trace)

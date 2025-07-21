from dataclasses import dataclass
from collections import defaultdict
from typing import Union

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

_YEAR_BUCKET_SIZE = 3
_EURO_BEGIN = "2024-06-14"
_TRAIN_START_YEAR = 2000


@dataclass
class Data:
    teams: ArrayLike
    goals: ArrayLike
    form: ArrayLike


@dataclass
class TeamData:
    home: Data
    away: Data
    tournament: ArrayLike
    year: ArrayLike


@dataclass
class Mappings:
    team: dict
    tournament: dict
    year: dict


@dataclass
class Dataset:
    train: TeamData
    test: TeamData
    mappings: Mappings


def preprocess(data: pd.DataFrame) -> Dataset:
    data = data[data["year"] > _TRAIN_START_YEAR].copy()
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    data["year_bucket"] = data["year"] // _YEAR_BUCKET_SIZE

    teams = sorted(set(data["home_team"]).union(set(data["away_team"])))
    tournaments = sorted(data["tournament"].unique())
    years = sorted(data["year_bucket"].unique())
    mappings = create_mappings(teams, tournaments, years)

    data["home_team_enc"] = data["home_team"].map(mappings.team)
    data["away_team_enc"] = data["away_team"].map(mappings.team)
    data["tournament_enc"] = data["tournament"].map(mappings.tournament)
    data["year_enc"] = data["year_bucket"].map(mappings.year)

    data = compute_recent_form(data)
    train, test = create_train_and_test_set(data)
    return Dataset(train=train, test=test, mappings=mappings)


def create_mappings(
    teams: ArrayLike, tournaments: ArrayLike, years: ArrayLike
) -> Mappings:
    team_mapping = {team: i for i, team in enumerate(teams)}
    tournament_mapping = {t: i for i, t in enumerate(tournaments)}
    year_mapping = {bucket: i for i, bucket in enumerate(years)}
    return Mappings(
        team=team_mapping, tournament=tournament_mapping, year=year_mapping
    )


def compute_recent_form(df, window=5):
    team_results = defaultdict(list)
    recent_win_rate = []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        h_id, a_id = row["home_team_enc"], row["away_team_enc"]
        h_results = team_results[h_id][-window:]
        a_results = team_results[a_id][-window:]

        h_winrate = sum(h_results) / len(h_results) if h_results else 0.5
        a_winrate = sum(a_results) / len(a_results) if a_results else 0.5
        recent_win_rate.append((h_winrate, a_winrate))

        result = np.sign(row["home_score"] - row["away_score"])
        team_results[h_id].append(1 if result == 1 else 0)
        team_results[a_id].append(1 if result == -1 else 0)

    df["home_form"], df["away_form"] = zip(*recent_win_rate)
    return df


def create_train_and_test_set(data: pd.DataFrame) -> Union[TeamData, TeamData]:
    train_set = data[data["date"] < _EURO_BEGIN]
    test_set = data[data["date"] > _EURO_BEGIN]
    train_home_data = Data(
        teams=train_set["home_team_enc"].values,
        goals=train_set["home_score"].values,
        form=train_set["home_form"].values,
    )
    test_home_data = Data(
        teams=test_set["home_team_enc"].values,
        goals=test_set["home_score"].values,
        form=test_set["home_form"].values,
    )
    train_away_data = Data(
        teams=train_set["away_team_enc"].values,
        goals=train_set["away_score"].values,
        form=train_set["away_form"].values,
    )
    test_away_data = Data(
        teams=test_set["away_team_enc"].values,
        goals=test_set["away_score"].values,
        form=test_set["away_form"].values,
    )
    train = TeamData(
        home=train_home_data,
        away=train_away_data,
        tournament=train_set["tournament_enc"].values,
        year=train_set["year_enc"].values,
    )
    test = TeamData(
        home=test_home_data,
        away=test_away_data,
        tournament=test_set["tournament_enc"].values,
        year=test_set["year_enc"].values,
    )
    return train, test

# Imports
import json
import os
import sys
from datetime import date, datetime

import polars as pl
import yaml

# Local imports
from app import app
from dash.dependencies import Input, Output, State
from layout.tabs import (
    make_div_h2h,
    make_div_serve_return,
    make_div_summary,
    make_div_under_pressure,
)

from atp_stats import Player, Renderer
from atp_stats.utils import get_player_name

# Load config
with open(os.getcwd() + "/webapp/config.yaml") as file:
    config = yaml.load(file, Loader=yaml.Loader)

serve_return_cols = config["serve_return_cols"]
under_pressure_cols = config["under_pressure_cols"]
tourney_level_map = config["tourney_level_map"]
rounds = config["rounds"]
details_mapping = config["details_mapping"]
matches_mapping = config["matches_mapping"]
h2h_mapping = config["h2h_mapping"]

# Load data
data_path = os.getcwd() + "/data"

all_matches = pl.read_parquet(data_path + "/matches.parquet")
all_players = pl.read_parquet(data_path + "/players.parquet")


def dt_to_iso(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


# CALLBACKS
@app.callback(
    [Output("open_state", "children"), Output("open_details", "children")],
    [Input("open_details", "n_clicks")],
    [State("open_state", "children")],
)
def toggle_collapse(n: int, is_open: bool):
    """open and closes window"""
    if n:
        open_det = "Close Description" if not is_open else "Open Description"
        return [not is_open, open_det]
    return [is_open, "Close Description"]


# Select Player and update all other filters field
@app.callback(
    [
        Output("player_matches", "data"),
        Output("player_info", "data"),
        Output("time_period", "min"),
        Output("time_period", "max"),
        Output("time_period", "marks"),
        Output("time_period", "value"),
        Output("surface", "value"),
        Output("surface", "options"),
        Output("tourney_level", "value"),
        Output("tourney_level", "options"),
        Output("tournament", "options"),
        Output("opponent", "options"),
        Output("round", "options"),
    ],
    [Input("player_name", "value")],
)
def select_player(player_name: str):
    """
    Given player name selects his matches, ranks and info and updates all the filters
    options, so that there is no incompatible displaying.
    """
    player = Player.from_raw_dataframes(player_name, all_matches, all_players)

    matches = player.matches
    info = player.info

    yr_min, yr_max = (
        matches.select([pl.min("year").alias("min"), pl.max("year").alias("max")])
        .to_numpy()
        .squeeze()
    )
    yr_min, yr_max = int(yr_min), int(yr_max)

    yr_marks = {
        i: {"label": str(i), "style": {"transform": "rotate(45deg)"}}
        for i in range(yr_min, yr_max + 1)
    }

    yr_value = [yr_min, yr_max]

    surfaces = tuple(matches.select([pl.col("surface").unique()]).to_series())
    surfaces_opt = [{"label": s, "value": s} for s in surfaces]

    tourney_levels = tuple(matches.select([pl.col("tourney_level").unique()]).to_series())
    tourney_levels_opt = [
        {"label": tourney_level_map[tl], "value": tl} for tl in tourney_levels
    ]

    tourney_opt = [
        {"label": t, "value": t}
        for t in matches.select([pl.col("tourney_name").unique()]).to_series().sort()
    ]
    opponents_opt = [
        {"label": o, "value": o}
        for o in matches.select([pl.col("opponent_name").unique()]).to_series().sort()
    ]

    rounds_opt = [
        r
        for r in rounds
        if r["value"] in matches.select([pl.col("round").unique()]).to_series()
    ]

    return [
        json.dumps(matches.to_dicts(), default=dt_to_iso),
        json.dumps(info, default=dt_to_iso),
        yr_min,
        yr_max,
        yr_marks,
        yr_value,
        surfaces,
        surfaces_opt,
        tourney_levels,
        tourney_levels_opt,
        tourney_opt,
        opponents_opt,
        rounds_opt,
    ]


@app.callback(
    Output("tab-content", "children"),
    [
        Input("tabs", "value"),
        Input("time_period", "value"),
        Input("surface", "value"),
        Input("tourney_level", "value"),
        Input("tournament", "value"),
        Input("opponent", "value"),
        Input("round", "value"),
        Input("opponent_rank", "value"),
    ],
    [
        State("player_name", "value"),
        State("player_matches", "data"),
        State("player_info", "data"),
    ],
)
def render_player(
    tab,
    time_period,
    surfaces,
    tourney_levels,
    tournaments,
    opponents,
    rounds,
    opponent_ranks,
    name,
    matches,
    info,
):
    """Main functionality to render a selected tab given current filters"""

    matches = pl.DataFrame(json.loads(matches))
    info = json.loads(info)

    player = Player(name, matches, info).filter(
        time_start=date(time_period[0], 1, 1),
        time_end=date(time_period[1], 1, 1),
        surfaces=surfaces,
        tourney_levels=tourney_levels,
        tournaments=tournaments,
        opponents=opponents,
        rounds=rounds,
        opponent_ranks=opponent_ranks,
    )

    renderer = Renderer(player)

    if tab == "summary":

        latest_matches = (
            player.selected_matches.sort(["tourney_date", "match_num"], reverse=True)
            .head(15)
            .with_columns(
                [
                    pl.col("tourney_level").apply(lambda v: tourney_level_map.get(v)),
                    pl.col("opponent_name").apply(get_player_name),
                ]
            )
            .select(list(matches_mapping.keys()))
            .rename(matches_mapping)
        )

        fig1, fig2 = renderer.plot_summary()

        info_to_use = (
            "player_name",
            "best_rank",
            "country_code",
            "birthdate",
            "height",
            "hand",
        )
        info_to_use_mp = [details_mapping.get(e) for e in info_to_use]
        info_data = sorted(
            [
                {"info": details_mapping.get(k), "value": v}
                for k, v in player.info.items()
                if k in info_to_use
            ],
            key=lambda d: info_to_use_mp.index(d["info"]),
        )

        stats_data = [
            {"info": k.replace("_", " ").replace("perc", "%"), "value": v}
            for k, v in player.perc_overall.to_dicts()[0].items()
        ]

        div = make_div_summary(fig1, fig2, info_data, stats_data, latest_matches)

    elif tab == "serve_return":

        fig = renderer.plot_serve_return_stats(columns=serve_return_cols)
        height = f"{400*len(serve_return_cols)}px"

        div = make_div_serve_return(fig, height)

    elif tab == "under_pressure":

        fig = renderer.plot_under_pressure(columns=under_pressure_cols)
        height = f"{500*len(under_pressure_cols)}px"

        div = make_div_under_pressure(fig, height)

    elif tab == "h2h":

        fig = renderer.plot_h2h()
        h2h = (
            player.h2h.select(h2h_mapping.keys())
            .rename(h2h_mapping)
            .with_columns([(pl.col(pl.Float64) * 100).round(2)])
        )

        div = make_div_h2h(fig, h2h)

    return div

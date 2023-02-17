import os
from typing import Tuple

import dash_bootstrap_components as dbc
import polars as pl
import yaml
from dash import dcc, html

with open(os.getcwd() + "/webapp/styles.yaml") as file:
    styles = yaml.safe_load(file)


style_col = styles["style_col"]
style_dropdown = styles["style_dropdown"]
style_top = styles["style_top"]
style_bottom = styles["style_bottom"]


def get_filter_rows(players: pl.DataFrame) -> Tuple[html.Div, html.Div, html.Div]:
    """
    Generates filter card
    """

    players_dropdown = dcc.Dropdown(
        id="player_name",
        placeholder="Choose a player",
        options=[
            {"label": str(n), "value": str(n)}
            for n in players.select(pl.col("player_name").unique()).to_series().sort()
        ],
        value="Roger Federer",
        clearable=False,
        style=style_dropdown,
    )

    surface_dropdown = dcc.Dropdown(
        id="surface", placeholder="Select surfaces", multi=True, style=style_dropdown
    )

    tourney_lvl_dropdown = dcc.Dropdown(
        id="tourney_level",
        placeholder="Select tournament types",
        multi=True,
        style=style_dropdown,
    )
    tourney_dropdown = dcc.Dropdown(
        id="tournament",
        placeholder="Select tournaments",
        searchable=True,
        multi=True,
        style=style_dropdown,
    )

    opponent_dropdown = dcc.Dropdown(
        id="opponent", placeholder="Select opponents", multi=True, style=style_dropdown
    )

    opponent_rank_dropdown = dcc.Dropdown(
        id="opponent_rank",
        placeholder="Select max opponent rank",
        options=[
            {"label": "Top 5", "value": 5},
            {"label": "Top 10", "value": 10},
            {"label": "Top 20", "value": 20},
            {"label": "Top 50", "value": 50},
            {"label": "Top 100", "value": 100},
        ],
        style=style_dropdown,
    )

    round_dropdown = dcc.Dropdown(
        id="round", placeholder="Select rounds", multi=True, style=style_dropdown
    )

    top_row = dbc.Row(
        id="top_row",
        style=style_top,
        children=[
            # Select Player
            dbc.Col(
                id="select_player",
                children=[
                    # html.H3("Player", style=style_h3),
                    players_dropdown
                ],
                width=3,
                style=style_col,
            ),
            # Select Surface
            dbc.Col(
                id="select_surface",
                children=[
                    # html.H3("Surface Type", style=style_h3),
                    surface_dropdown
                ],
                width=3,
                style=style_col,
            ),
            # Select Tournament Level
            dbc.Col(
                id="select_tourney_lvl",
                children=[
                    # html.H3("Tournament Levels", style=style_h3),
                    tourney_lvl_dropdown,
                ],
                width=3,
                style=style_col,
            ),
            # Select Tournaments
            dbc.Col(
                id="select_tourney",
                children=[
                    # html.H3("Select Tournaments", style=style_h3),
                    tourney_dropdown,
                ],
                width=3,
                style=style_col,
            ),
        ],
    )

    bottom_row = dbc.Row(
        id="bottom_row",
        style=style_bottom,
        children=[
            # Select Opponents
            dbc.Col(
                id="select_opponent",
                children=[
                    # html.H3("Select Opponents", style=style_h3),
                    opponent_dropdown
                ],
                width=3,
                style=style_col,
            ),
            # Select Top Rank
            dbc.Col(
                id="select_opponent_rank",
                children=[
                    # html.H3("Opponents Rank", style=style_h3),
                    opponent_rank_dropdown,
                ],
                width=3,
                style=style_col,
            ),
            # Select Rounds
            dbc.Col(
                id="select_round",
                children=[
                    # html.H3("Rounds", style=style_h3),
                    round_dropdown
                ],
                width=3,
                style=style_col,
            ),
            dbc.Col(
                id="select_period",
                children=[
                    # html.H3("Time Period", style=style_h3),
                    html.H3("", style={"margin-top": "15px"}),
                    dcc.RangeSlider(id="time_period", step=2),
                ],
                width=3,
                style=style_col,
            ),
        ],
    )

    filters = dbc.Card(
        id="filters",
        children=[
            dbc.CardHeader("Filters"),
            dbc.CardBody(
                [
                    top_row,
                    bottom_row,
                    # row3
                ]
            ),
        ],
        style={"margin-left": "1%", "margin-right": "1%"},
    )
    return filters

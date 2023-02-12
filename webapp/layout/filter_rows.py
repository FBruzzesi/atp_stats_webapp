import os
from typing import Tuple

import dash_bootstrap_components as dbc
import polars as pl
import yaml
from dash import dcc, html

with open(os.getcwd() + "/webapp/styles.yaml") as file:
    styles = yaml.load(file, Loader=yaml.Loader)

style_h3 = styles["style_h3"]
style_dropdown = styles["style_dropdown"]
style_row1 = styles["style_row1"]
style_row2 = styles["style_row2"]
style_row3 = styles["style_row3"]


def get_filter_rows(players: pl.DataFrame) -> Tuple[html.Div, html.Div, html.Div]:
    """
    Generates filter card
    """

    players_dropdown = dcc.Dropdown(
        id="player_name",
        options=[
            {"label": str(n), "value": str(n)}
            for n in players.select(pl.col("player_name").unique()).to_series().sort()
        ],
        value="Roger Federer",
        clearable=False,
        style=style_dropdown,
    )

    surface_dropdown = dcc.Dropdown(id="surface", multi=True, style=style_dropdown)
    tourney_lvl_dropdown = dcc.Dropdown(
        id="tourney_level", multi=True, style=style_dropdown
    )
    tourney_dropdown = dcc.Dropdown(
        id="tournament", searchable=True, multi=True, style=style_dropdown
    )

    opponent_dropdown = dcc.Dropdown(id="opponent", multi=True, style=style_dropdown)
    opponent_rank_dropdown = dcc.Dropdown(
        id="opponent_rank",
        options=[
            {"label": "Top 5", "value": 5},
            {"label": "Top 10", "value": 10},
            {"label": "Top 20", "value": 20},
            {"label": "Top 50", "value": 50},
            {"label": "Top 100", "value": 100},
        ],
        style=style_dropdown,
    )
    round_dropdown = dcc.Dropdown(id="round", multi=True, style=style_dropdown)

    col_style = {"display": "inline-block", "margin-left": "1%", "width": "23.5%"}

    row1 = dbc.Row(
        id="row1",
        style=style_row1,
        children=[
            # Select Player
            dbc.Col(
                id="select_player",
                children=[html.H3("Player", style=style_h3), players_dropdown],
                width=3,
                style=col_style,
            ),
            # Select Surface
            dbc.Col(
                id="select_surface",
                children=[html.H3("Surface Type", style=style_h3), surface_dropdown],
                width=3,
                style=col_style,
            ),
            # Select Tournament Level
            dbc.Col(
                id="select_tourney_lvl",
                children=[
                    html.H3("Tournament Levels", style=style_h3),
                    tourney_lvl_dropdown,
                ],
                width=3,
                style=col_style,
            ),
            # Select Tournaments
            dbc.Col(
                id="select_tourney",
                children=[
                    html.H3("Select Tournaments", style=style_h3),
                    tourney_dropdown,
                ],
                width=3,
                style=col_style,
            ),
        ],
    )

    row2 = dbc.Row(
        id="row2",
        style=style_row2,
        children=[
            # Select Opponents
            dbc.Col(
                id="select_opponent",
                children=[html.H3("Select Opponents", style=style_h3), opponent_dropdown],
                width=3,
                style=col_style,
            ),
            # Select Top Rank
            dbc.Col(
                id="select_opponent_rank",
                children=[
                    html.H3("Opponents Rank", style=style_h3),
                    opponent_rank_dropdown,
                ],
                width=3,
                style=col_style,
            ),
            # Select Rounds
            dbc.Col(
                id="select_round",
                children=[html.H3("Rounds", style=style_h3), round_dropdown],
                width=3,
                style=col_style,
            ),
        ],
    )

    row3 = dbc.Row(
        id="row3",
        style=style_row3,
        children=[
            dbc.Col(
                id="select_period",
                children=[
                    html.H3("Time Period", style=style_h3),
                    dcc.RangeSlider(id="time_period", step=1),
                ],
                style={"display": "inline-block", "margin-left": "1%", "width": "47%"},
                width=6,
            )
        ],
    )

    filters = dbc.Card(
        id="filters",
        children=[
            dbc.CardHeader("Filters"),
            dbc.CardBody([row1, row2, row3]),
        ],
        style={"margin-left": "1%", "margin-right": "1%"},
    )
    return filters

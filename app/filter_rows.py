import os

import yaml
from dash import dcc, html

with open(os.getcwd() + "/utils/styles.yaml") as file:
    styles = yaml.load(file, Loader=yaml.Loader)

style_h3 = styles["style_h3"]
style_dropdown = styles["style_dropdown"]
style_row1 = styles["style_row1"]
style_row2 = styles["style_row2"]
style_row3 = styles["style_row3"]


def get_filter_rows(matches_df, players_df):

    row1 = html.Div(
        id="row1",
        className="row",
        style=style_row1,
        children=[
            # Select Player
            html.Div(
                id="select_player",
                className="three columns",
                style={"display": "inline-block", "margin-left": "1%", "width": "23.5%"},
                children=[
                    html.H3("Player", style=style_h3),
                    dcc.Dropdown(
                        id="player_name",
                        options=[
                            {"label": n, "value": n}
                            for n in sorted(players_df["player_name"].unique())
                        ],
                        value="Roger Federer",
                        clearable=False,
                        style=style_dropdown,
                    ),
                ],
            ),
            # Select Surface
            html.Div(
                id="select_surface",
                className="three columns",
                style={"display": "inline-block", "margin-left": "1%", "width": "23.5%"},
                children=[
                    html.H3("Surface Type", style=style_h3),
                    dcc.Dropdown(id="surface", multi=True, style=style_dropdown),
                ],
            ),
            # Select Tournament Level
            html.Div(
                id="select_tourney_lvl",
                className="three columns",
                style={"display": "inline-block", "margin-left": "1%", "width": "23.5%"},
                children=[
                    html.H3("Tournament Levels", style=style_h3),
                    dcc.Dropdown(id="tourney_level", multi=True, style=style_dropdown),
                ],
            ),
            # Select Tournaments
            html.Div(
                id="select_tourney",
                className="three columns",
                style={"display": "inline-block", "margin-left": "1%", "width": "23.5%"},
                children=[
                    html.H3("Select Tournaments", style=style_h3),
                    dcc.Dropdown(
                        id="tournament", searchable=True, multi=True, style=style_dropdown
                    ),
                ],
            ),
        ],
    )

    row2 = html.Div(
        id="row2",
        className="row",
        style=style_row2,
        children=[
            # Select Opponents
            html.Div(
                id="select_opponent",
                className="three columns",
                style={"display": "inline-block", "margin-left": "1%", "width": "23.5%"},
                children=[
                    html.H3("Select Opponents", style=style_h3),
                    dcc.Dropdown(id="opponent", multi=True, style=style_dropdown),
                ],
            ),
            # Select Top Rank
            html.Div(
                id="select_opponent_rank",
                className="three columns",
                style={"display": "inline-block", "margin-left": "1%", "width": "23.5%"},
                children=[
                    html.H3("Opponents Rank", style=style_h3),
                    dcc.Dropdown(
                        id="opponent_rank",
                        options=[
                            {"label": "Top 5", "value": 5},
                            {"label": "Top 10", "value": 10},
                            {"label": "Top 20", "value": 20},
                            {"label": "Top 50", "value": 50},
                            {"label": "Top 100", "value": 100},
                        ],
                        style=style_dropdown,
                    ),
                ],
            ),
            # Select Rounds
            html.Div(
                id="select_round",
                className="three columns",
                style={"display": "inline-block", "margin-left": "1%", "width": "23.5%"},
                children=[
                    html.H3("Rounds", style=style_h3),
                    dcc.Dropdown(id="round", multi=True, style=style_dropdown),
                ],
            ),
        ],
    )

    row3 = html.Div(
        id="row3",
        className="row",
        style=style_row3,
        children=[
            html.Div(
                id="select_period",
                className="six columns",
                style={"display": "inline-block", "margin-left": "1%", "width": "47%"},
                children=[
                    html.H3("Time Period", style=style_h3),
                    dcc.RangeSlider(id="time_period", step=1),
                ],
            )
        ],
    )

    return row1, row2, row3

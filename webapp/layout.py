import os

import dash_bootstrap_components as dbc
import polars as pl
from dash import dcc, html
from filter_rows import get_filter_rows

intro = open("webapp/intro.md", "r").read()
data_path = os.getcwd() + "/data"

matches = pl.read_parquet(data_path + "/matches.parquet")
players = pl.read_parquet(data_path + "/players.parquet")

tab_style = {"fontWeight": "bold"}

# Github link button
github = dbc.Button(
    [html.I(className="bi bi-github"), " Github"],
    outline=True,
    href="https://github.com/FBruzzesi/atp_stats_webapp",
    id="gh-link",
    external_link=True,
    # style=social_style,
)

# Report a Bug link button
report_bug = dbc.Button(
    [html.I(className="bi bi-bug-fill"), " Report a Bug"],
    outline=True,
    href="https://github.com/FBruzzesi/atp_stats_webapp/issues",
    id="bug-link",
    external_link=True,
    # style=social_style,
)

# Linkedin link button
linkedin = dbc.Button(
    [html.I(className="bi bi-linkedin"), " Linkedin"],
    outline=True,
    href="https://linkedin.com/in/francesco-bruzzesi/",
    id="linkedin-link",
    external_link=True,
    # style=social_style,
)

# Support link button
support = dbc.Button(
    [html.I(className="bi bi-cup-fill"), " Buy me a coffee"],
    outline=True,
    href="https://ko-fi.com/francescobruzzesi",
    id="support-link",
    external_link=True,
    # style=social_style,
)

social_container = dbc.Container(
    id="social", children=[github, report_bug, linkedin, support]
)
# Header Container
header = html.Div(
    id="app-header",
    children=[
        dbc.Row(
            children=[
                dbc.Col(
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    [
                                        "ATP Statistics ",
                                        html.I(className="bi bi-pencil-square"),
                                    ]
                                )
                            ],
                            id="app-title",
                        )
                    ],
                    align="center",
                    width={"offset": 4},
                    style={"margin-top": 20},
                ),
                dbc.Col(
                    children=social_container,
                    align="center",
                    width={"size": 5, "offset": 7},
                    style={"margin-top": -35},
                ),
            ],
        )
    ],
)


layout = html.Div(
    [
        html.Details(
            open=True,
            children=[
                html.Summary(
                    id="open_details",
                    children="Close Description",
                    style={"margin-left": "1.5%"},
                ),
                dcc.Markdown(intro, style={"margin-left": "3%", "margin-top": "10pt"}),
                html.Div(id="open_state", children=True, style={"display": "none"}),
            ],
        ),
        # Hidden Div Block
        html.Div(
            [
                # Store selected player matches data
                html.Div(id="selected_player_matches", style={"display": "none"}),
                # Store selected player details data
                html.Div(id="selected_player_details", style={"display": "none"}),
                # Store selected player ranking data
                html.Div(id="selected_player_rank", style={"display": "none"}),
            ],
            style={"display": "none"},
        ),
        *get_filter_rows(players),
        html.Div(
            className="row",
            style={"margin-left": "2%", "margin-right": "2%", "margin-top": "1%"},
            children=[
                dcc.Tabs(
                    id="tabs",
                    value="summary",
                    children=[
                        dcc.Tab(
                            label="Player Summary",
                            value="summary",
                            style=tab_style,
                            selected_style=tab_style,
                        ),
                        dcc.Tab(
                            label="Serve & Return",
                            value="serve_return",
                            style=tab_style,
                            selected_style=tab_style,
                        ),
                        dcc.Tab(
                            label="Under Pressure",
                            value="under_pressure",
                            style=tab_style,
                            selected_style=tab_style,
                        ),
                        dcc.Tab(
                            label="H2H",
                            value="h2h",
                            style=tab_style,
                            selected_style=tab_style,
                        ),
                    ],
                    colors={
                        "border": "white",
                        "primary": "gold",
                        "background": "cornsilk",
                    },
                ),
                html.Div(id="tab-content"),
            ],
        ),
    ]
)

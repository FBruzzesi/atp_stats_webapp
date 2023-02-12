import os
from functools import partial

import dash_bootstrap_components as dbc
import polars as pl
from dash import dcc, html

from .filter_rows import get_filter_rows

intro = open("webapp/layout/intro.md", "r").read()
data_path = os.getcwd() + "/data"

players = pl.read_parquet(data_path + "/players.parquet")

tab_style = {"fontWeight": "bold"}

make_tab = partial(dcc.Tab, style=tab_style, selected_style=tab_style)

attribution = html.Details(
    title="Description",
    open=True,
    style={"margin-top": "1%", "margin-left": "1.5%"},
    children=[
        html.Summary(id="open_details", children="Close Description"),
        html.Div(id="open_state", children=True, style={"display": "none"}),
        dbc.Card(
            id="description",
            children=[
                dbc.CardHeader("Data Attribution and Usage"),
                dbc.CardBody([dcc.Markdown(intro)]),
            ],
        ),
    ],
)

store_matches = dcc.Store(id="player_matches")
store_info = dcc.Store(id="player_info")

tabs = dbc.Row(
    [
        dcc.Tabs(
            id="tabs",
            value="summary",
            children=[
                make_tab(label="Player Summary", value="summary"),
                make_tab(label="Serve & Return", value="serve_return"),
                make_tab(label="Under Pressure", value="under_pressure"),
                make_tab(
                    label="H2H",
                    value="h2h",
                ),
            ],
            colors={
                "border": "white",
                "primary": "gold",
                "background": "cornsilk",
            },
        ),
    ],
)


page = html.Div(
    [
        attribution,
        store_matches,
        store_info,
        *get_filter_rows(players),
        tabs,
        dbc.Row(id="tab-content"),  #  html.Div(id="tab-content"),
    ]
)

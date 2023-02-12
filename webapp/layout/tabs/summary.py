from typing import Any, Dict, List

import plotly.graph_objects as go
import polars as pl
from dash import dash_table, dcc, html


def make_div(
    fig1: go.Figure,
    fig2: go.Figure,
    info_data: List[Dict[str, Any]],
    stats_data: List[Dict[str, Any]],
    latest_matches: pl.DataFrame,
) -> html.Div:

    """Create div for summary tab"""
    dtable_info = dash_table.DataTable(
        data=info_data,
        columns=[{"id": c, "name": c, "type": "datetime"} for c in ("info", "value")],
        style_cell={"text-align": "left"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
        style_header={"display": "none"},
    )

    dtable_stats = dash_table.DataTable(
        data=stats_data,
        columns=[{"id": c, "name": c} for c in ("info", "value")],
        style_cell_conditional=[{"if": {"column_id": "info"}, "text-align": "left"}],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
        style_header={"display": "none"},
    )

    dtable_recent = dash_table.DataTable(
        data=latest_matches.to_dicts(),
        columns=[{"id": c, "name": c} for c in latest_matches.columns],
        style_cell_conditional=[{"if": {"column_id": "info"}, "text-align": "left"}],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
        style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
    )

    div = html.Div(
        [
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="two columns",
                        children=[
                            html.H5(
                                "Player Details",
                                style={"text-align": "center", "margin-bottom": "3%"},
                            ),
                            dtable_info,
                        ],
                        style={
                            "margin-top": "5%",
                            "margin-left": "5%",
                            "width": "40%",
                            "display": "inline-block",
                        },
                    ),
                    html.Div(
                        className="two columns",
                        children=[
                            html.H5("Player Statistics", style={"text-align": "center"}),
                            dtable_stats,
                        ],
                        style={
                            "margin-top": "1.5%",
                            "margin-left": "10%",
                            "width": "40%",
                            "display": "inline-block",
                        },
                    ),
                ],
            ),
            html.Div(
                className="row",
                children=[
                    dcc.Graph(
                        figure=fig1,
                        id="graph_summary",
                        hoverData={"points": [{"customdata": "Japan"}]},
                    ),
                    dcc.Graph(
                        figure=fig2,
                        id="graph_summary",
                        hoverData={"points": [{"customdata": "Japan"}]},
                    ),
                ],
                style={"margin-top": "2%"},
            ),
            # This Row is left unaffected
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="eleven columns",
                        children=[
                            html.H3("Recent Matches", style={"text-align": "center"}),
                            dtable_recent,
                        ],
                        style={"margin-top": "1%", "margin-left": "3%"},
                    )
                ],
            ),
        ]
    )

    return div

import plotly.graph_objects as go
import polars as pl
from dash import dash_table, dcc, html


def make_div(fig: go.Figure, h2h: pl.DataFrame) -> html.Div:
    """Create div for h2h stats"""

    dtable = dash_table.DataTable(
        data=h2h.to_dicts(),
        columns=[{"id": c, "name": c} for c in h2h.columns],
        sort_action="native",
        style_cell_conditional=[{"if": {"column_id": "Opponent"}, "text-align": "left"}],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
        style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
        page_size=20,
    )

    div = html.Div(
        children=[
            dcc.Graph(
                figure=fig,
                id="h2h_graph",
                hoverData={"points": [{"customdata": "Japan"}]},
                style={"height": "95%"},
            ),
            html.Div(
                [html.H3("H2H Statistics", style={"text-align": "center"}), dtable],
                style={"margin-top": "2%", "margin-left": "5%"},
            ),
        ],
        style={"width": "95%", "display": "inline-block", "padding": "0 20"},
    )
    return div

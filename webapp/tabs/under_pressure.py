import plotly.graph_objects as go
from dash import dcc, html


def make_div(fig: go.Figure, height: str) -> html.Div:
    """Create div for under pressure stats"""

    div = html.Div(
        children=[
            dcc.Graph(
                figure=fig,
                id="graph_under_pressure",
                hoverData={"points": [{"customdata": "Japan"}]},
                style={"height": "95%"},
            )
        ],
        style={
            "height": height,
            "width": "95%",
            "display": "inline-block",
            "padding": "0 20",
        },
    )
    return div

from dash import html, dcc 
import plotly.graph_objects as go

def make_serve_return_div(fig: go.Figure, height: str) -> html.Div:
    """Create div for serve and return stats"""

    div = html.Div(
        children=[
            dcc.Graph(
                figure=fig,
                id="graph_serve_return",
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
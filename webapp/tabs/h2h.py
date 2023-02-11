"""
div = html.Div(
        children=[
            dcc.Graph(
                figure=fig_h2h,
                id="h2h_graph",
                hoverData={"points": [{"customdata": "Japan"}]},
                style={"height": "95%"},
            ),
            html.Div(
                [html.H3("H2H Statistics", style={"text-align": "center"}), dt_h2h],
                style={"margin-top": "2%", "margin-left": "5%"},
            ),
        ],
        style={"width": "95%", "display": "inline-block", "padding": "0 20"},
    )
"""
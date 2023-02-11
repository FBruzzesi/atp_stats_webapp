"""html.Div(
            children=[
                dcc.Graph(
                    figure=fig_under_pressure,
                    id="graph_under_pressure",
                    hoverData={"points": [{"customdata": "Japan"}]},
                    style={"height": "95%"},
                )
            ],
            style={
                "height": f"{500*len(under_pressure_cols)}px",
                "width": "95%",
                "display": "inline-block",
                "padding": "0 20",
            },
        )"""
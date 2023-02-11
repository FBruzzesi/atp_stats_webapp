from dash import dcc, html
import plotly.graph_objects as go

def make_summary_div(fig1, fig2, ) -> html.Div:

    # Generate datatables
    details = tp.player_details
    details.columns = ["info", "value"]
    details["info"] = details["info"].map(details_mapping)

    stats = pd.DataFrame(
        data={
            "info": tp.perc_overall.index,
            "value": np.round(tp.perc_overall.to_numpy(), 2),
        }
    )
    stats["info"] = ["% " + c.split("_")[-1].capitalize() for c in stats["info"]]

    matches = (
        tp.selected_matches.sort_values(
            ["tourney_date", "match_num"], ascending=[False, False]
        )
        .assign(
            tourney_level=lambda x: x["tourney_level"].map(tourney_level_map),
            opponent_name=lambda x: x["opponent_name"].apply(get_player_name),
        )
        .loc[:, matches_mapping.keys()]
        .rename(columns=matches_mapping)
    )

    dt_details = dash_table.DataTable(
        data=details.astype(str).to_dict("records"),
        columns=[{"id": c, "name": c, "type": "datetime"} for c in details.columns],
        style_cell={"text-align": "left"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
        style_header={"display": "none"},
    )

    dt_stats = dash_table.DataTable(
        data=stats.to_dict("records"),
        columns=[{"id": c, "name": c} for c in stats.columns],
        style_cell_conditional=[{"if": {"column_id": "info"}, "text-align": "left"}],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
        style_header={"display": "none"},
    )

    dt_recent = dash_table.DataTable(
        data=matches.head(15).to_dict("records"),
        columns=[{"id": c, "name": c} for c in matches.columns],
        style_cell_conditional=[{"if": {"column_id": "info"}, "text-align": "left"}],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
        ],
        style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
    )

    div = html.Div(
        [
            # First 2 rows got "broken" from css className not working
            # html.Div(
            #     className='row',
            #     children=[
            #         html.Div(
            #             className='two columns',
            #             children=[
            #                 html.H5('Player Details', style={'text-align': 'center', 'margin-bottom': '3%'}),
            #                 dt_details
            #             ],
            #             style={'margin-top': '5%', 'margin-left': '5%'}
            #         ),
            #         html.Div(
            #             className='nine columns',
            #             children=[
            #                 dcc.Graph(
            #                     figure=fig1,
            #                     id='graph_summary',
            #                     hoverData={'points': [{'customdata': 'Japan'}]},
            #                 ),
            #             ],
            #             style={'margin-top': '2%'}
            #         ),
            #     ]
            # ),
            # html.Div(
            #     className='row',
            #     children=[
            #         html.Div(
            #             className='nine columns',
            #             children=[
            #                 dcc.Graph(
            #                     figure=fig2,
            #                     id='graph_summary',
            #                     hoverData={'points': [{'customdata': 'Japan'}]},
            #                 ),
            #             ],
            #             style={'margin-top': '-1%', 'width': '75%', 'display': 'inline-block'}
            #         ),
            #         html.Div(
            #             className='two columns',
            #             children=[
            #                 html.H5('Player Statistics', style={'text-align': 'center'}),
            #                 dt_stats
            #             ],
            #             style={'margin-top': '1.5%',}
            #         ),
            #     ]
            # ),
            # Fast fix from here to ln 295
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
                            dt_details,
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
                            html.H5(
                                "Player Statistics", style={"text-align": "center"}
                            ),
                            dt_stats,
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
                            dt_recent,
                        ],
                        style={"margin-top": "1%", "margin-left": "3%"},
                    )
                ],
            ),
        ]
    )

    return div
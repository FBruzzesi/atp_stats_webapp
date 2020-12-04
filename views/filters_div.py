import dash_core_components as dcc
import dash_html_components as html

def get_filters_div(matches_df):
    # First Div Block
    row1 = html.Div([
            # Select Player
        html.Div([
            html.H5('Select a Player', style={'justify': 'center', 'align': 'center', 'text-align': 'center'}),
            dcc.Dropdown(
                id='player_name',
                options=[{'label': n, 'value':n} for n in sorted(matches_df['player_name'].unique())],
                value='Roger Federer',
                clearable=False,
                style={'justify': 'center', 'align': 'center', 'text-align': 'center'}
                ),
            ],
            style={'width': '24%', 'display': 'inline-block', 'margin-left': '1%'}
        ),
        # Select Surface
        html.Div([
            html.H5('Select Surface Type', style={'justify': 'center', 'align': 'center', 'text-align': 'center'}),
            dcc.Dropdown(
                id='surface',
                multi=True, 
                style={'justify': 'center', 'align': 'center', 'text-align': 'center'}
                ),
            ],
            style={'width': '24%', 'display': 'inline-block', 'margin-left': '1%'}
        ),
        # Select Tournament Level
        html.Div([
            html.H5('Select Tournament Levels', style={'justify': 'center', 'align': 'center', 'text-align': 'center'}),
            dcc.Dropdown(
                id='tourney_level',
                multi=True,
                style={'justify': 'center', 'align': 'center', 'text-align': 'center'}
                ),
            ],
            style={'width': '24%', 'display': 'inline-block', 'margin-left': '1%'}
        ),
        # Select Tournaments
        html.Div([
            html.H5('Select Tournaments', style={'justify': 'center', 'align': 'center', 'text-align': 'center'}),
            dcc.Dropdown(
                id='tournament',
                searchable=True, 
                multi=True,
                style={'justify': 'center', 'align': 'center', 'text-align': 'center'}
                ),
            ],
            style={'width': '24%', 'display': 'inline-block', 'margin-left': '1%'}
        ),
        
    ], style={'backgroundColor': 'rgb(250, 250, 250)', 'padding': '10px 5px'}
    )


    row2 = html.Div([
        # Select Opponents
        html.Div([
            html.H5('Select Opponents', style={'justify': 'center', 'align': 'center', 'text-align': 'center'}),
            dcc.Dropdown(
                id='opponent',
                multi=True,
                style={'justify': 'center', 'align': 'center', 'text-align': 'center'}
                )
            ],
            style={'width': '24%', 'display': 'inline-block', 'margin-left': '1%'},
        ),
        # Select Top Rank
        html.Div([
            html.H5('Select Opponents Rank', style={'justify': 'center', 'align': 'center', 'text-align': 'center'}),
            dcc.Dropdown(
                id='opponent_rank',
                options=[{'label': 'Top 5', 'value': 5}, 
                            {'label': 'Top 10', 'value': 10}, 
                            {'label': 'Top 20', 'value': 20}, 
                            {'label': 'Top 50', 'value': 50}, 
                            {'label': 'Top 100', 'value': 100}
                        ],
                style={'justify': 'center', 'align': 'center', 'text-align': 'center'}
                )
            ],
            style={'width': '24%', 'display': 'inline-block', 'margin-left': '1%'},
        ),
        # Select Rounds
        html.Div([
            html.H5('Select Rounds', style={'justify': 'center', 'align': 'center', 'text-align': 'center'}),
            dcc.Dropdown(
                id='round',
                multi=True,
                #clearable=False,
                style={'justify': 'center', 'align': 'center', 'text-align': 'center'}
                )
            ],
            style={'width': '24%', 'display': 'inline-block', 'margin-left': '1%'},
        ),
        
    ], style={'backgroundColor': 'rgb(250, 250, 250)', 'padding': '10px 5px'}
    )

    row3 = html.Div([
        html.Div([
            html.H5('Select Time Period', style={'justify': 'center', 'align': 'center', 'text-align': 'center'}),
            dcc.RangeSlider(
                id='time_period',
                step=1
                )
            ], 
            style={'width': '50%', 'display': 'inline-block', 'margin-left': '1%'}
        )
    ],style={'borderBottom': 'thin lightgrey solid', 'backgroundColor': 'rgb(250, 250, 250)', 'padding': '10px 5px 35px'}
    )


    return row1, row2, row3
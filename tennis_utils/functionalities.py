import dash_core_components as dcc
import dash_html_components as html

def get_filters_div(matches_df, players_df):
    # First Div Block
    row1 = html.Div(
        id='row1',
        className='row',
        style={'borderTop': 'thin lightgrey solid', 'backgroundColor': 'rgb(250, 250, 250)', 'padding': '10px 5px'},
        children=[
        # Select Player
        html.Div(
            id='select_player',
            className='three columns',
            style={'display': 'inline-block', 'margin-left': '1%'},
            children=[
                html.Strong('Player', style={'margin-left': '45%', 'fontSize': '18px', 'color': 'mediumblue', 'font-family': 'Arial'}),
                dcc.Dropdown(
                    id='player_name',
                    options=[{'label': n, 'value':n} for n in sorted(players_df['player_name'].unique())],
                    value='Roger Federer',
                    clearable=False,
                    style={'justify': 'center', 'align': 'center', 'text-align': 'center', 'margin-top': '1%', 'height': '50px'}
                ),
            ],
        ),
        # Select Surface
        html.Div(
            id='select_surface',
            className='three columns',
            style={'display': 'inline-block', 'margin-left': '1%'},
            children=[
                html.Strong('Surface Type', style={'margin-left': '40%', 'fontSize': '18px', 'color': 'mediumblue', 'font-family': 'Arial'}),
                dcc.Dropdown(
                    id='surface',
                    multi=True,
                    style={'justify': 'center', 'align': 'center', 'text-align': 'center', 'margin-top': '1%', 'height': '50px'}
                ),
            ],
        ),
        # Select Tournament Level
        html.Div(
            id='select_tourney_lvl',
            className='three columns',
            style={'display': 'inline-block', 'margin-left': '1%'},
            children=[
                html.Strong('Tournament Levels', style={'margin-left': '30%', 'fontSize': '18px', 'color': 'mediumblue', 'font-family': 'Arial'}),
                dcc.Dropdown(
                    id='tourney_level',
                    multi=True,
                    style={'justify': 'center', 'align': 'center', 'text-align': 'center', 'margin-top': '1%', 'height': '50px'}
                ),
            ],
        ),
        # Select Tournaments
        html.Div(
            id='select_tourney',
            className='three columns',
            style={'display': 'inline-block', 'margin-left': '1%'},
            children=[
                html.Strong('Select Tournaments', style={'margin-left': '30%', 'fontSize': '18px', 'color': 'mediumblue', 'font-family': 'Arial'}),
                dcc.Dropdown(
                    id='tournament',
                    searchable=True, 
                    multi=True,
                    style={'justify': 'center', 'align': 'center', 'text-align': 'center', 'margin-top': '1%', 'height': '50px'}
                ),
            ],
        ),
    ])


    row2 = html.Div(
        id='row2',
        className='row',
        style={'backgroundColor': 'rgb(250, 250, 250)', 'padding': '20px 5px'},
        children=[
        # Select Opponents
        html.Div(
            id='select_opponent',
            className='three columns',
            style={'display': 'inline-block', 'margin-left': '1%'},
            children=[
                html.Strong('Select Opponents', style={'margin-left': '30%', 'fontSize': '18px', 'color': 'mediumblue', 'font-family': 'Arial'}),
                dcc.Dropdown(
                    id='opponent',
                    multi=True,
                    style={'justify': 'center', 'align': 'center', 'text-align': 'center', 'margin-top': '1%', 'height': '50px'}
                )
            ],
        ),
        # Select Top Rank
        html.Div(
            id='select_opponent_rank',
            className='three columns',
            style={'display': 'inline-block', 'margin-left': '1%'},
            children=[
                html.Strong('Opponents Rank', style={'margin-left': '30%', 'fontSize': '18px', 'color': 'mediumblue', 'font-family': 'Arial'}),
                dcc.Dropdown(
                    id='opponent_rank',
                    options=[{'label': 'Top 5', 'value': 5}, 
                             {'label': 'Top 10', 'value': 10}, 
                             {'label': 'Top 20', 'value': 20}, 
                             {'label': 'Top 50', 'value': 50}, 
                             {'label': 'Top 100', 'value': 100}
                            ],
                    style={'justify': 'center', 'align': 'center', 'text-align': 'center', 'margin-top': '1%', 'height': '50px'}
                    )
            ],
        ),
        # Select Rounds
        html.Div(
            id='select_round',
            className='three columns',
            style={'display': 'inline-block', 'margin-left': '1%'},
            children=[
                html.Strong('Rounds',  style={'margin-left': '40%', 'fontSize': '18px', 'color': 'mediumblue', 'font-family': 'Arial'}),
                dcc.Dropdown(
                    id='round',
                    multi=True,
                    style={'justify': 'center', 'align': 'center', 'text-align': 'center', 'margin-top': '1%', 'height': '50px'}
                )
            ],
        ),
        
    ], 
    )

    row3 = html.Div(
        id='row3',
        className='row',
        style={'borderBottom': 'thin lightgrey solid', 'backgroundColor': 'rgb(250, 250, 250)', 'padding': '10px 5px 35px'},
        children=[
        html.Div(
            id='select_period',
            className='six columns',
            style={'display': 'inline-block', 'margin-left': '1%'},
            children=[
                html.Strong('Time Period', style={'margin-left': '45%', 'fontSize': '18px', 'color': 'mediumblue', 'font-family': 'Arial'}),
                dcc.RangeSlider(
                    id='time_period',
                    step=1
                )
            ], 
        )
    ], 
    )


    return row1, row2, row3
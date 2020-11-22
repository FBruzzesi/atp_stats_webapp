# Imports
import pandas as pd
import numpy as np

from datetime import date, datetime as dt
import os, re
import plotly.graph_objects as go, plotly.express as px, plotly.figure_factory as ff
from plotly.offline import init_notebook_mode
from plotly.subplots import make_subplots

import dash, dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from tennis_utils.player import TennisPlayer, TennisDataLoader, TennisPlayerDataLoader
from tennis_utils.settings import tennis_player_settings
from views.filters_div import get_filters_div

surface_colors = tennis_player_settings['surface_colors']
colors = tennis_player_settings['colors']
# Data Load
data_path = os.getcwd()+'/data'

tdl = TennisDataLoader(data_path+'/matches.parquet', data_path+'/players.parquet')
matches_df, players_df = tdl.matches, tdl.players


row1, row2, row3 = get_filters_div(matches_df)



tourney_level_map = {
    'A': 'Atp 500',
    'M': 'Master 1000',
    'G': 'Grand Slam',
    'F': 'Finals', 
    'C': 'Challenger',
    'D': 'Davis Cup'
}




# Initialize App and define its layout
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    # Hidden Div Block
    html.Div([
        # Store selected player matches data
        html.Div(id='selected_player_matches', style={'display': 'none'}),
        # Store selected player details data
        html.Div(id='selected_player_details', style={'display': 'none'}),
        # Store selected player ranking data
        html.Div(id='selected_player_rank', style={'display': 'none'})
        ], style={'display': 'none'}
    ),
    row1,
    row2,
    row3,
    html.Div([
        dcc.Tabs(id='tabs', 
            value='sr_dist', 
            children=[
                dcc.Tab(label='Player Details', value='details'),
                dcc.Tab(label='Serve & Return - Time Series', value='sr_ts'),
                dcc.Tab(label='Serve & Return - Distributions', value='sr_dist'),
                dcc.Tab(label='Under Pressure', value='pressure'),
            ], 
            colors={
                'border': 'white',
                'primary': 'gold',
                'background': 'cornsilk'
                }
        ),
        html.Div(id='tab-content')
    ]),
])




'''
Defining App callbacks i.e. interactions
'''


# Select Player and update all other filters field
@app.callback(
    [
     Output(component_id='selected_player_matches', component_property='children'),
     Output(component_id='selected_player_details', component_property='children'),
     Output(component_id='selected_player_rank', component_property='children'),
     Output(component_id='time_period', component_property='min'),
     Output(component_id='time_period', component_property='max'),
     Output(component_id='time_period', component_property='marks'),
     Output(component_id='time_period', component_property='value'),
     Output(component_id='surface', component_property='value'),
     Output(component_id='surface', component_property='options'),
     Output(component_id='tourney_level', component_property='value'),
     Output(component_id='tourney_level', component_property='options'),
     Output(component_id='tournament', component_property='options'),
     Output(component_id='opponent', component_property='options'),
     Output(component_id='round', component_property='options')
    ],
    [Input(component_id='player_name', component_property='value')]
)
def select_player(player_name):

    tpdl = TennisPlayerDataLoader(player_name, matches_df, players_df)
    # Subset selected player matches data
    player_matches = tpdl.player_matches
    player_details = tpdl.player_details
    player_rank = tpdl.player_rank

    yr_min, yr_max = player_matches['year'].min(), player_matches['year'].max()
    yr_marks = {(i): {'label': str(i), 'style': {'transform':'rotate(45deg)'}}  for i in range(yr_min, yr_max+1, 1)}

    #yr_marks={i: str(i)  for i in range(yr_min, yr_max+1, 1)}
    yr_value = [yr_min, yr_max]

    surfaces = player_matches['surface'].unique().tolist()
    surfaces_opt = [{'label': s, 'value': s } for s in surfaces]

    tourney_levels = player_matches['tourney_level'].unique().tolist()
    tourney_levels_opt = [{'label': tourney_level_map[tl], 'value': tl} for tl in tourney_levels]

    tourney_opt = [{'label': t, 'value': t} for t in sorted(player_matches['tourney_name'].unique())]
    opponents_opt = [{'label': o, 'value': o} for o in sorted(player_matches['opponent_name'].unique())]


    rounds = [
        {'label': 'Final', 'value': 'F'},
        {'label': 'Semifinal', 'value': 'SF'},
        {'label': 'Quarterfinal', 'value': 'QF'},
        {'label': 'R16', 'value': 'R16'},
        {'label': 'R32', 'value': 'R32'},
        {'label': 'R64', 'value': 'R64'}, 
        {'label': 'R128', 'value': 'R128'}, 
        {'label': 'Q1', 'value': 'Q1'},
        {'label': 'Q2', 'value': 'Q2'},
        {'label': 'Q3', 'value': 'Q3'},
        {'label': 'Round Robin', 'value': 'RR'}
    ]
    player_rounds = player_matches['round'].unique() #['round'].unique()
    rounds_opt = [r for r in rounds if r['value'] in player_rounds]

    return [player_matches.to_json(date_format='iso'), 
            player_details.to_json(date_format='iso'), 
            player_rank.to_json(date_format='iso'), 
            yr_min, yr_max, yr_marks, yr_value, surfaces, surfaces_opt, 
            tourney_levels, tourney_levels_opt, tourney_opt, opponents_opt, rounds_opt]
    







@app.callback(
    Output(component_id='tab-content', component_property='children'),
    [
     Input(component_id='tabs', component_property='value'),
     Input(component_id='selected_player_matches', component_property='children'),
     Input(component_id='selected_player_details', component_property='children'),
     Input(component_id='selected_player_rank', component_property='children'),
     Input(component_id='time_period', component_property='value'),
     Input(component_id='surface', component_property='value'),
     Input(component_id='tourney_level', component_property='value'),
     Input(component_id='tournament', component_property='value'),
     Input(component_id='opponent', component_property='value'),
     Input(component_id='round', component_property='value'),
     Input(component_id='opponent_rank', component_property='value')
    ],
    [State(component_id='player_name', component_property='value')]
)
    
def render_player(tab,
                  player_matches,
                  player_details,
                  player_rank,
                  time_period,
                  surfaces,
                  tourney_levels,
                  tournaments,
                  opponents,
                  rounds,
                  opponent_ranks,
                  player_name
                  ):

    y1, y2 = time_period
    time_start, time_end = date(y1, 1, 1), date(y2, 12, 31)

    tp = TennisPlayer(
            player_name = player_name,
            player_matches = pd.read_json(player_matches),
            player_rank = pd.read_json(player_rank),
            player_details = pd.read_json(player_details),
            time_start = time_start,
            time_end = time_end,
            surfaces = surfaces,
            tourney_levels = tourney_levels,
            tournaments = tournaments,
            opponents = opponents,
            rounds = rounds,
            opponent_ranks = opponent_ranks
    )
    

    if tab == 'details':

        # Generate figures
        fig_rank = tp.plot_rank()
        fig_yearly_winrate = tp.plot_yearly_wr()
        fig_winrate = tp.plot_winrate()
        fig_surface_wl = tp.plot_surface_wl()

        # Create html Div
        div = html.Div([
            dcc.Graph(
                figure=fig_rank,
                id='rnk',
                hoverData={'points': [{'customdata': 'Japan'}]}
            ),
            dcc.Graph(
                figure=fig_yearly_winrate,
                id='yearly_winrate',
                hoverData={'points': [{'customdata': 'Japan'}]},
                style={'display':'inline-block', 'width':'75%'}
            ),
            dcc.Graph(
                figure=fig_winrate,
                id='winrate',
                hoverData={'points': [{'customdata': 'Japan'}]},
                style={'display':'inline-block', 'width':'25%'}
            ),
            dcc.Graph(
                figure=fig_surface_wl,
                id='winrate_by_surface',
                hoverData={'points': [{'customdata': 'Japan'}]},
            ),
            
        ], style={'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )



    elif tab == 'sr_ts':
    
        fig_cols_overtime = tp.plot_cols_overtime()

        div = html.Div([
            dcc.Graph(
                figure=fig_cols_overtime,
                id='col_timeseries',
                hoverData={'points': [{'customdata': 'Japan'}]},
                style={'height': '95%'}
            )
        ], style={'height': '1600px', 'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )

    elif tab == 'sr_dist':


        fig_cols_distribution = tp.plot_cols_distribution()

        div = html.Div([
            dcc.Graph(
                figure=fig_cols_distribution,
                id='col_distrib',
                hoverData={'points': [{'customdata': 'Japan'}]},
                style={'height': '95%'}
            )
        ], style={'height': '1600px', 'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )

    elif tab == 'pressure':

        pass
    
    
    return div


    
    

'''
@app.callback(
    [
    ],
    [Input(component_id='player_name', component_property='value'),
     Input(component_id='column', component_property='value'),
     
     Input(component_id='tourney_level', component_property='value'),
     Input(component_id='surface', component_property='value'),
     Input(component_id='opponents', component_property='value'),]
)
def stats_graphs(player_name, col, period_interval, tourney_levels, surfaces, opponents):
    
    
    
    matches = mdata[mdata['name'] == player_name]
    pdetails = pdata[pdata['name'] == player_name]

    tp = RenderTennisPlayer(player_name=player_name, matches=matches, player_details=pdetails, 
                            
                            surfaces=surfaces,
                            tourney_levels=tourney_levels, 
                            opponents=opponents
                           )
        
    
    color = colors[0]
    
    
    fig_win_rate = tp.plot_yearly_wr()
    fig_rank = tp.plot_rank()
    fig_surface_wl = tp.plot_surface_wl()
    fig_col_overtime = tp.plot_col_overtime(col, color)
    fig_surface_boxplot = tp.plot_surface_boxplot(col)
    fig_col_distplot = tp.plot_col_distplot(col, colors)
    
    
    return fig_rank, fig_win_rate, fig_surface_wl, fig_col_overtime, fig_surface_boxplot, fig_col_distplot #, generate_table(tp.player_details)
'''

#app.run_server(mode="inline")
#app.run_server(mode='jupyterlab')
app.run_server(debug=True)













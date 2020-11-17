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

from tennis_utils.scrapers import SackmanScraper
from tennis_utils.player import TennisPlayer, TennisDataLoader, TennisPlayerDataLoader

from tennis_utils.settings import tennis_player_settings

from views.filters_div import render_filters

surface_colors = tennis_player_settings['surface_colors']
colors = tennis_player_settings['colors']
# Data Load
data_path = os.getcwd()+'/data'

tdl = TennisDataLoader(data_path+'/matches.parquet', data_path+'/players.parquet')
matches_df, players_df = tdl.matches, tdl.players


row1, row2, row3 = render_filters(matches_df)
# tdata['tourney_name'] = np.where(tdata['tourney_name'].str.startswith('Davis Cup'), 'Davis Cup', tdata['tourney_name'])

# unq_tdata = mdata[['tourney_name', 'tourney_level', 'surface']].drop_duplicates()

# Settings

surface_colors = {
    'Clay':'firebrick', 
    'Grass':'seagreen', 
    'Hard':'midnightblue', 
    'Carpet':'limegreen'
}

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
        dcc.Tabs(id="tabs-styled-with-inline", 
            value='tab-1', 
            children=[
                dcc.Tab(label='Tab 1', value='tab-1'),
                dcc.Tab(label='Tab 2', value='tab-2'),
                dcc.Tab(label='Tab 3', value='tab-3'),
            ], 
            colors={
                "border": "white",
                "primary": "gold",
                "background": "cornsilk"
            }
    ),
    html.Div(id='tabs-content-inline')
    ]),
    html.Div([
        dcc.Graph(
            id='rnk',
            hoverData={'points': [{'customdata': 'Japan'}]}
        ),
        dcc.Graph(
            id='win_rate',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
        
    ], style={'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
    ),
    html.Div([
        dcc.Graph(
            id='sunburst',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
    ),
    html.Div([
        dcc.Graph(
            id='lineplot',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '95%', 'height':'700px', 'display': 'inline-block', 'padding': '0 20'}
    ),
    html.Div([
        dcc.Graph(id='boxplot', style={'display': 'inline-block', 'width': '45%'}),
        dcc.Graph(id='distplot', style={'display': 'inline-block', 'width': '45%'})
    ])
])




'''
Defining App callbacks i.e. interactions
'''

@app.callback(
    [Output(component_id='selected_player_matches', component_property='children'),
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
    player_matches = tpdl.player_matches #matches_df[matches_df['player_name'] == player_name]
    player_details = tpdl.player_details #players_df[players_df['player_name']==player_name]
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
    [
     Output(component_id='rnk', component_property='figure'),
     Output(component_id='win_rate', component_property='figure'),
     Output(component_id='sunburst', component_property='figure'),
     Output(component_id='lineplot', component_property='figure'),
     Output(component_id='boxplot', component_property='figure'),
     Output(component_id='distplot', component_property='figure'),
    ],
    [
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
    
def render_player(player_matches,
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

        
    
    color = colors[0]
    
    col = 'perc1stIn'
    fig_win_rate = tp.plot_yearly_wr()
    fig_rank = tp.plot_rank()
    fig_surface_wl = tp.plot_surface_wl()
    fig_col_overtime = tp.plot_col_overtime(col, color)
    fig_surface_boxplot = tp.plot_surface_boxplot(col)
    fig_col_distplot = tp.plot_col_distplot(col, colors)
    
    
    return fig_rank, fig_win_rate, fig_surface_wl, fig_col_overtime, fig_surface_boxplot, fig_col_distplot


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













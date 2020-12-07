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

from tennis_utils.player import TennisPlayerRenderer, TennisDataLoader, TennisPlayerDataLoader
from tennis_utils.settings import tennis_player_settings
from views.filters_div import get_filters_div

surface_colors = tennis_player_settings['surface_colors']
colors = tennis_player_settings['colors']
# Data Load
data_path = os.getcwd()+'/data'

tdl = TennisDataLoader(data_path+'/matches.parquet', data_path+'/players.parquet')
matches_df, players_df = tdl.matches, tdl.players


# row1, row2, row3 = get_filters_div(matches_df)



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
    *get_filters_div(matches_df),
    html.Div([
        dcc.Tabs(id='tabs', 
            value='summary', 
            children=[
                dcc.Tab(label='Player Summary', value='summary'),
                dcc.Tab(label='Serve & Return - Time Series', value='service_return'),
                dcc.Tab(label='Serve & Return - Distributions', value='service_return_distribution'),
                dcc.Tab(label='Under Pressure', value='pressure'),
                dcc.Tab(label='H2H', value='h2h'),
            ], 
            colors={
                'border': 'white',
                'primary': 'gold',
                'background': 'cornsilk'
                },
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

    tp = TennisPlayerRenderer(
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
    

    if tab == 'summary':

        # Generate figures
        fig_summary = tp.plot_summary()

        details_mapping = {
            'player_name': 'Player Name',
            'best_rank': 'Best Rank',
            'country_code': 'Nationality',
            'birthdate': 'Birthdate',
            'age': 'Age', 
            'hand':'Hand',
            'height':'Height'
        }
        details = tp.player_details
        details.columns=['info', 'value']
        details['info'] = details['info'].map(details_mapping)

        stats = pd.DataFrame(data={
                    'info': tp.perc_overall.index,
                    'value': np.round(tp.perc_overall.to_numpy(),2)
                })
        stats['info'] = [ '% ' + c.split('_')[-1].capitalize() for c in stats['info']]

        dt_details = dash_table.DataTable(
            data=details.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in details.columns],
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
            style_header = {'display': 'none'}
        )

        dt_stats = dash_table.DataTable(
            data=stats.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in stats.columns],
            sort_action='native',
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
            style_header = {'display': 'none'}
        )


        # Create html Div
        div = html.Div([
            dcc.Graph(
                figure=fig_summary,
                id='graph_summary',
                hoverData={'points': [{'customdata': 'Japan'}]}
            ),
            html.Div([
                html.Div([
                    html.H3('Player Details', style={'text-align': 'center'}),
                    dt_details
                    ], style={'margin-top': '2%', 'margin-left': '5%', 'width':'30%', 'display': 'inline-block'}),
                 html.Div([
                     html.H3('Player Statistics', style={'text-align': 'center'}),
                     dt_stats
                    ], style={'margin-top': '2%', 'margin-left': '30%', 'width':'30%', 'display': 'inline-block'})
                ])
        ], style={'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )





    elif tab == 'service_return':
        
        cols_to_plot = ['firstIn', 'firstWon', 'secondWon', 'returnWon', 'ace', 'df']
        fig_service_return = tp.plot_stats(columns=cols_to_plot)

        div = html.Div([
            dcc.Graph(
                figure=fig_service_return,
                id='graph_service_return',
                hoverData={'points': [{'customdata': 'Japan'}]},
                style={'height': '95%'}
            )
        ], style={'height': f'{400*len(cols_to_plot)}px', 'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )

    elif tab == 'service_return_distribution':

        cols_to_plot = ['firstIn', 'firstWon', 'secondWon', 'returnWon']
        fig_distribution = tp.plot_distribution(columns=cols_to_plot)

        div = html.Div([
            dcc.Graph(
                figure=fig_distribution,
                id='graph_distribution',
                hoverData={'points': [{'customdata': 'Japan'}]},
                style={'height': '95%'}
            )
        ], style={'height': f'{400*len(cols_to_plot)}px', 'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )

    elif tab == 'pressure':
        
        cols_to_plot = ['bpSaved', 'bpConverted', 'tbWon', 'decidingSetWon']
        fig_pressure = tp.plot_stats(columns=cols_to_plot)

        div = html.Div([
            dcc.Graph(
                figure=fig_pressure,
                id='pressure_graph',
                hoverData={'points': [{'customdata': 'Japan'}]},
                style={'height': '95%'}
            )
        ], style={'height': f'{400*len(cols_to_plot)}px', 'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )
    
    elif tab == 'h2h':

        fig_h2h = tp.plot_h2h()

        col_mapping = {
            'opponent_name': 'Opponent', 
            'matches_played': 'Matches Played', 
            'matches_won': 'Matches Won', 
            'win_rate': 'Winrate',
            'perc_ace': '% Aces',
            'perc_df': '% Double Faults',
            'perc_firstIn': '% First In',
            'perc_firstWon': '% First Won', 
            'perc_secondWon': '% Second Won', 
            'perc_returnWon': '% Return Won', 
            'perc_bpConverted': '% BP Converted',
            'perc_bpSaved': '% BP Saved', 
            'perc_tbWon': '% TB Won',
            'perc_decidingSetWon': '% Deciding Sets Won'
        }

        h2h = tp.h2h.round(2)[col_mapping.keys()].rename(columns = col_mapping)


        dt = dash_table.DataTable(
            data=h2h.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in h2h.columns],
            sort_action='native', filter_action='native',
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            page_size=20
        )

        div = html.Div([
            dcc.Graph(
                figure=fig_h2h,
                id='h2h_graph',
                hoverData={'points': [{'customdata': 'Japan'}]},
                style={'height': '95%'}
            ),
            html.Div(dt, style={'margin-top': '2%', 'margin-left': '5%'})
        ], style={'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )

    
    return div


    
    

app.run_server(debug=True)













# Imports
import pandas as pd
import numpy as np

from datetime import date, datetime as dt
import os, re, yaml

import dash, dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# Local imports 
from app import app
from utils.tennis import TennisDataLoader, PlayerDataLoader, PlayerRenderer, get_player_name

# Load config
with open(os.getcwd() + '/utils/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.Loader)

surface_colors = config['surface_colors']
colors = config['colors']
serve_return_cols = config['serve_return_cols']
under_pressure_cols = config['under_pressure_cols']
tourney_level_map = config['tourney_level_map']
rounds = config['rounds']
details_mapping = config['details_mapping']
matches_mapping = config['matches_mapping']
h2h_mapping = config['h2h_mapping']

# Load data
data_path = os.getcwd() + '/data'

tdl = TennisDataLoader(data_path=data_path)
matches_df, players_df = tdl.matches, tdl.players


# CALLBACKS
@app.callback(
    [Output('open_state', 'children'),
    Output('open_details', 'children')],
    [Input('open_details', 'n_clicks')],
    [State('open_state', 'children')],
)
def toggle_collapse(n, is_open):
    if n:
        open_det = 'Close Description' if not is_open else 'Open Description'
        return [not is_open, open_det]
    return [is_open, 'Close Description']

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

    tpdl = PlayerDataLoader(player_name, matches_df, players_df)
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
    
def render_player(
        tab,
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

    tp = PlayerRenderer(
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

        # Generate summary graph
        fig1, fig2 = tp.plot_summary()

        # Generate datatables
        details = tp.player_details
        details.columns=['info', 'value']
        details['info'] = details['info'].map(details_mapping)

        stats = pd.DataFrame(data={
                    'info': tp.perc_overall.index,
                    'value': np.round(tp.perc_overall.to_numpy(),2)
                })
        stats['info'] = [ '% ' + c.split('_')[-1].capitalize() for c in stats['info']]

        matches = (tp.selected_matches
                    .sort_values(['tourney_date', 'match_num'], ascending=[False, False])
                    .assign(
                        tourney_level = lambda x: x['tourney_level'].map(tourney_level_map),
                        opponent_name = lambda x: x['opponent_name'].apply(get_player_name)
                        )
                    .loc[:, matches_mapping.keys()]
                    .rename(columns=matches_mapping)
        )

        dt_details = dash_table.DataTable(
            data=details.astype(str).to_dict('records'),
            columns=[{'id': c, 'name': c, 'type': 'datetime'} for c in details.columns],
            style_cell={'text-align': 'left'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
            style_header = {'display': 'none'},
        )

        dt_stats = dash_table.DataTable(
            data=stats.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in stats.columns],
            style_cell_conditional=[{'if': {'column_id': 'info'}, 'text-align': 'left'}],
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
            style_header = {'display': 'none'}
        )

        dt_recent = dash_table.DataTable(
            data=matches.head(15).to_dict('records'),
            columns=[{'id':c, 'name':c} for c in matches.columns],
            style_cell_conditional=[{'if': {'column_id': 'info'}, 'text-align': 'left'}],
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        )

        div = html.Div([
                html.Div(
                    className='row',
                    children=[
                        html.Div(
                            className='two columns',
                            children=[
                                html.H5('Player Details', style={'text-align': 'center', 'margin-bottom': '3%'}),
                                dt_details
                            ],
                            style={'margin-top': '5%', 'margin-left': '5%'}
                        ),
                        html.Div(
                            className='nine columns',
                            children=[
                                dcc.Graph(
                                    figure=fig1,
                                    id='graph_summary',
                                    hoverData={'points': [{'customdata': 'Japan'}]},
                                ),
                            ],
                            style={'margin-top': '2%'} 
                        ),
                    ]
                ),
                html.Div(
                    className='row',
                    children=[
                        html.Div(
                            className='nine columns',
                            children=[
                                dcc.Graph(
                                    figure=fig2,
                                    id='graph_summary',
                                    hoverData={'points': [{'customdata': 'Japan'}]},
                                ),
                            ],
                            style={'margin-top': '-1%'} 
                        ),
                        html.Div(
                            className='two columns',
                            children=[
                                html.H5('Player Statistics', style={'text-align': 'center'}),
                                dt_stats
                            ],
                            style={'margin-top': '1.5%',}
                        ),
                    ]
                ),
                html.Div(
                    className='row',
                    children=[
                        html.Div(
                            className='eleven columns',
                            children=[
                                html.H3('Recent Matches', style={'text-align': 'center'}),
                                dt_recent
                            ],
                            style={'margin-top': '1%', 'margin-left': '3%'}
                        )
                    ],
                ),
            ])


    elif tab == 'serve_return':
        
        fig_serve_return = tp.plot_serve_return_stats(columns=serve_return_cols)

        div = html.Div(
                children=[
                    dcc.Graph(
                        figure=fig_serve_return,
                        id='graph_serve_return',
                        hoverData={'points': [{'customdata': 'Japan'}]},
                        style={'height': '95%'}
                    )
                ], 
                style={'height': f'{400*len(serve_return_cols)}px', 'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )


    elif tab == 'under_pressure':
        
        fig_under_pressure = tp.plot_under_pressure(columns=under_pressure_cols)

        div = html.Div(
                children=[
                    dcc.Graph(
                        figure=fig_under_pressure,
                        id='graph_under_pressure',
                        hoverData={'points': [{'customdata': 'Japan'}]},
                        style={'height': '95%'}
                    )
                ],
                style={'height': f'{500*len(under_pressure_cols)}px', 'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
            )
    
    elif tab == 'h2h':

        fig_h2h = tp.plot_h2h()

        h2h = (tp.h2h
                .loc[:, h2h_mapping.keys()]
                .rename(columns = h2h_mapping)
                )

        h2h.iloc[:, 3:] = (100*h2h.iloc[:, 3:]).round(2)

        dt_h2h = dash_table.DataTable(
            data=h2h.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in h2h.columns],
            sort_action='native', filter_action='native',
            style_cell_conditional=[{'if': {'column_id': 'Opponent'}, 'text-align': 'left'}],
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            page_size=20
        )

        div = html.Div(
                children=[
                    dcc.Graph(
                        figure=fig_h2h,
                        id='h2h_graph',
                        hoverData={'points': [{'customdata': 'Japan'}]},
                        style={'height': '95%'}
                    ),
                    html.Div([
                        html.H3('H2H Statistics', style={'text-align': 'center'}),
                        dt_h2h
                        ], style={'margin-top': '2%', 'margin-left': '5%'})
                ],
                style={'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
            )
    
    return div
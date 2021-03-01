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
from tennis_utils.functionalities import get_filters_div

import yaml

with open(os.getcwd() + '/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.Loader)

surface_colors = config['surface_colors']
colors = config['colors']
serve_return_cols = config['serve_return_cols']
under_pressure_cols = config['under_pressure_cols']
tourney_level_map = config['tourney_level_map']
rounds = config['rounds']
details_mapping = config['details_mapping']
h2h_mapping = config['h2h_mapping']


# Data Load
data_path = os.getcwd() + '/data'

tdl = TennisDataLoader(data_path=data_path)
matches_df, players_df = tdl.matches, tdl.players




# Initialize App and define its layout
app = dash.Dash(__name__, 
        title='ATP Stats',
        external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        
server = app.server




header=html.Div([
    html.Div(
        html.H1(
            children='ATP Statistics', # Title line
            style={'textAlign': 'center'},
            className='content-container'
            )
    )],
    className='header',
)


markdown = """
**Data Attribution:** The data used here is (part of) the amazing dataset created by [**Jeff Sackmann**](http://www.jeffsackmann.com/) 
(Check out his [github repository](https://github.com/JeffSackmann/tennis_atp))

**Data Usage:** In particular, I am using atp singles from 1995 to 2020 data. I am currently working towards an independent data gathering solution.

**Bug Fix:** This is a MVP which I had fun developing, mostly on weekends, for personal use. Therefore I am sure it is possible to find bugs and non-working interactions. 
If you find any or just want to get in touch with me, please feel free to reach out by [Linkedin](https://www.linkedin.com/in/francesco-bruzzesi/)

**How it Works:** I hope this is straightforward; down below there are a series of possible filters you want to play with. Everything is based upon a selected player, in the sense that only such player statistics will appear. Then:

- _Player Summary_: Shows rank, rank points, winrate over time and a set of overall statistics as well as some player information.
- _Serve & Return_: Shows serve and return statistics over time with a 95% confidence interval and distribution of all selected matches.
- _Under Pressure_: Shows under pressure statistics over time with a 95% confidence interval.
- _H2H_: Head-to-Head, shows winrate againsts most played opponents.

If you'd like to support this project, you can do so by [buying me a coffee](https://www.buymeacoffee.com/fbruzzesi)
"""

app.layout = html.Div([
    header,
    html.Hr(style={'width': '96%', 'margin-top': '1%', 'margin-bottom': '1%'}),
    html.Details(
        open=True,
        children=[
        html.Summary(id='open_details', children = 'Close Description', style={'margin-left': '1.5%'}),
        dcc.Markdown(markdown, style={'margin-left': '3%', 'margin-top': '10pt'}),
        html.Div(id='open_state', children=True, style={'display': 'none'})
    ]),
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
                dcc.Tab(label='Serve & Return', value='serve_return'),
                dcc.Tab(label='Under Pressure', value='under_pressure'),
                dcc.Tab(label='H2H', value='h2h'),
            ], 
            colors={
                'border': 'white',
                'primary': 'gold',
                'background': 'cornsilk'
                },
        ),
        html.Div(id='tab-content')
    ], style={'margin-top':'1%'}),
])


'''
Defining App callbacks i.e. interactions
'''

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

        # Generate summary graph
        fig_summary = tp.plot_summary()

        # Generate datatables
        details = tp.player_details
        details.columns=['info', 'value']
        details['info'] = details['info'].map(details_mapping)

        stats = pd.DataFrame(data={
                    'info': tp.perc_overall.index,
                    'value': np.round(tp.perc_overall.to_numpy(),2)
                })
        stats['info'] = [ '% ' + c.split('_')[-1].capitalize() for c in stats['info']]


        dt_details = dash_table.DataTable(
            data=details.astype(str).to_dict('records'),
            columns=[{'id': c, 'name': c, 'type': 'datetime'} for c in details.columns],
            style_cell={'textAlign': 'left'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
            style_header = {'display': 'none'}
        )

        dt_stats = dash_table.DataTable(
            data=stats.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in stats.columns],
            sort_action='native',
            style_cell_conditional=[{'if': {'column_id': 'info'}, 'textAlign': 'left'}],
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
                    ], style={'margin-top': '2%', 'margin-left': '35%', 'width':'25%', 'display': 'inline-block'})
                ])
        ], style={'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )





    elif tab == 'serve_return':
        
        fig_serve_return = tp.plot_serve_return_stats(columns=serve_return_cols)

        div = html.Div([
            dcc.Graph(
                figure=fig_serve_return,
                id='graph_serve_return',
                hoverData={'points': [{'customdata': 'Japan'}]},
                style={'height': '95%'}
            )
        ], style={'height': f'{400*len(serve_return_cols)}px', 'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )


    elif tab == 'under_pressure':
        
        fig_under_pressure = tp.plot_under_pressure(columns=under_pressure_cols)

        div = html.Div([
            dcc.Graph(
                figure=fig_under_pressure,
                id='graph_under_pressure',
                hoverData={'points': [{'customdata': 'Japan'}]},
                style={'height': '95%'}
            )
        ], style={'height': f'{500*len(under_pressure_cols)}px', 'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )
    
    elif tab == 'h2h':

        fig_h2h = tp.plot_h2h()

        

        h2h = tp.h2h[h2h_mapping.keys()].rename(columns = h2h_mapping)
        h2h.iloc[:, 3:] = (100*h2h.iloc[:, 3:]).round(2)

        dt_h2h = dash_table.DataTable(
            data=h2h.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in h2h.columns],
            sort_action='native', filter_action='native',
            style_cell_conditional=[{'if': {'column_id': 'Opponent'}, 'textAlign': 'left'}],
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
            html.Div([
                html.H3('H2H Statistics', style={'text-align': 'center'}),
                dt_h2h
                ], style={'margin-top': '2%', 'margin-left': '5%'})
        ], style={'width': '95%', 'display': 'inline-block', 'padding': '0 20'}
        )

    
    return div


    
if __name__ == '__main__':
    app.run_server(debug=True)
#app.run_server(debug=True)













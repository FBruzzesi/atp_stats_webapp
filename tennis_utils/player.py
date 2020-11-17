import pandas as pd
import numpy as np
import dask
from dask import delayed
import os, re, warnings, multiprocessing as mp

from datetime import date, datetime as dt

from typing import List, Set, Tuple, Dict, Optional

import plotly.graph_objects as go, plotly.express as px, plotly.figure_factory as ff
from plotly.offline import init_notebook_mode
from plotly.subplots import make_subplots

# warnings.filterwarnings('ignore')

from tennis_utils.settings import tennis_player_settings
surface_colors = tennis_player_settings['surface_colors']


def timer(f, *args, **kwargs):
    '''
    Timer decorator for functions
    '''
    def wrapper(*args, **kwargs):
        tic = dt.now()
        result = f(*args, **kwargs)
        toc = dt.now()
        print(f'@{f.__name__} took {toc-tic}')
        return result
    return wrapper
    

def get_player_name(name):

    return '. '.join(['.'.join([e[0] for e in name.split(' ')[:-1]]), name.split(' ')[-1]])



class TennisDataLoader:
    '''
    Loads tennis matches data and players details data given both paths and stores them in 
    self.matches and self.players

    Attributes
    ----------
    self.matches: pd.DataFrame
        dataframe containing all the players matches
    self.players: pd.DataFrame
        dataframe containing all the players details
    self.matches_path: str
        path where self.matches is read from
    self.players_path: str
        path where self.players is read from
    '''

    def __init__(self, matches_data_path: str, players_data_path: str, type='parquet', sep=','):
        '''
        Loads and stores matches and players data

        Parameters
        ----------
        matches_data_path: str
            path of matches data
        players_data_path: str
            path of players data
        type: str, default 'parquet' 
            Type of extention: one between 'parquet' or 'csv'. 
        sep: str, default ','
            Field delimiter for the input files if type='csv'
        '''
        
        if type == 'parquet':

            self.matches = pd.read_parquet(matches_data_path)
            self.players = pd.read_parquet(players_data_path)

        elif type == 'csv':
            self.matches = pd.read_csv(matches_data_path, sep = sep)
            self.players = pd.read_csv(players_data_path, sep = sep)

        else:
            raise Exception('Can only load parquet and csv format')

        self.matches_path = matches_data_path
        self.players_path = players_data_path


    def __repr__(self):
        
        n_matches = self.matches.shape[0]
        n_players = self.players.shape[0]

        return f'TennisDataLoader storing {n_matches} matches and {n_players} players data'




class TennisPlayerDataLoader:
    '''
    Create static attributes of a given tennis player, namely

    Attributes
    ----------
    player_name: str
        Tennis player name
    matches: pd.DataFrame
        Dataframe containing all matches the player played in his/her career
    n_matches: int
        Number of matches the player played in his/her career
    rank_df: pd.DataFrame
        Dataframe containing time series of player rank and rank points over time
    player_details: pd.DataFrame
        Dataframe containing player details

    Methods
    -------
    get_rank
        Calculates time series of player rank and rank points
    get_player_details
        Retrieves player details from matches and players information dataframes
    
    '''
    def __init__(self, player_name: str, player_matches: pd.DataFrame, player_details: pd.DataFrame):
        '''
        Parameters
        ----------
        player_name: str
            Tennis player name
        tdl: TennisDataLoader
            TennisDataLoader instance
        '''
        self.player_name = player_name
        self.player_matches = player_matches.loc[player_matches['player_name']==player_name]
        self.n_matches = self.player_matches.shape[0]


        self.player_rank = self.get_rank()

        player_details = player_details[player_details['player_name']==player_name]
        self.player_details = self.get_player_details(player_details)


    def get_rank(self):
        '''
        Calculate time series of player rank and rank points
        
        Returns
        -------
        rank_df: pd.DataFrame
            rank and rank points over time
        '''
        
        rank_df = (self.player_matches[['tourney_date', 'rank', 'rank_points']]
                       .drop_duplicates().sort_values('tourney_date')
                       .dropna(subset=['rank', 'rank_points'])
        )

        rank_df[['rank', 'rank_points']] = rank_df[['rank', 'rank_points']].astype(int)

        return rank_df


    def get_player_details(self, player_details):
        '''
        Retrieves player details

        Returns
        -------
        player_details: pd.DataFrame
            Dataframe containing player details
        '''
        age = np.round((date.today() - player_details['birthdate'].astype('datetime64[ns]').dt.date).dt.days/365.25, 2)

        player_details = (player_details.assign(age = age)
                                        .assign(hand = self.player_matches['hand'].max())
                                        .assign(height = self.player_matches['ht'].max(skipna=False))
                                        .assign(best_rank = self.player_rank['rank'].min())
                                        .assign(most_rank_pts = self.player_rank['rank_points'].max())
        )

        return player_details




class TennisPlayer:
    '''
    Attributes
    ----------
    player_name: str
        Name of the tennis player
    player_matches: pd.DataFrame
        Dataframe containing all player matches
    player_details: pd.DataFrame
        Dataframe containing player information
    filters: Dict
        Dictionary containing filters to select matches
    selected_matches: pd.DataFrame
        Dataframe containing matches after applying filters
    n_matches: int
        Number of selected_matches
    player_rank: pd.DataFrame
        Dataframe containing player ranking time series
    

    Methods
    -------
    '''
    def __init__(self, 
                 player_name: str,
                 player_matches: pd.DataFrame,
                 player_rank: pd.DataFrame,
                 player_details: pd.DataFrame,
                 time_start: Optional[date] = None,
                 time_end: Optional[date] = None,
                 surfaces: Optional[List[str]] = None,
                 tourney_levels: Optional[List[str]] = None,
                 tournaments: Optional[List[str]] = None,
                 opponents: Optional[List[str]] = None,
                 rounds: Optional[List[str]] = None,
                 opponent_ranks: Optional[int] = None,
                 ):

        
        self.player_name = player_name
        self.player_matches = player_matches
        self.player_details = player_details
        
        self.filters = {
            'time_start': time_start,
            'time_end': time_end,
            'surface': surfaces,
            'tourney_level': tourney_levels,
            'tournament': tournaments,
            'opponent_name': opponents,
            'round': rounds,
            'opponent_rank': opponent_ranks
        }
        

        self.selected_matches = self.select_matches()
        self.n_matches = self.selected_matches.shape[0]
        self.player_rank = self.get_rank(player_rank)

        self.yearly_wr = self.get_yearly_winrate()
        self.surface_wl = self.get_surface_winloss()
            

    def __repr__(self):
        
        return f'{self.player_name}, number of matches: {self.n_matches}'
    

    def select_matches(self):
        '''
        Subsets self.matches based on time period, surfaces, tournament levels, 
        tournaments, opponents, rounds and n_top_seeds

        '''

        time_start = self.filters['time_start'] if self.filters['time_start'] is not None else date(1970,1,1)
        time_end = self.filters['time_end'] if self.filters['time_end'] is not None else date(2999,12,31)
        time_mask = pd.to_datetime(self.player_matches['tourney_date']).dt.date.between(time_start, time_end).to_numpy()

        masks = [time_mask]
        

        for key, value in self.filters.items():

            if value is not None and key not in ['time_start', 'time_end']:

                m = self.player_matches['opponent_rank'] < value if key=='n_top_seed' else self.player_matches[key].isin(value)

                masks.append(m)

        return self.player_matches.loc[np.all(np.array(masks).T, axis=1)]
        

    def get_rank(self, rank_df):
        '''
        Calculate time series of player rank and rank points
        
        Returns
        -------
        rank_df: pd.DataFrame
            rank and rank points over time
        '''
        
        time_start = self.filters['time_start'] if self.filters['time_start'] is not None else date(1970,1,1)
        time_end = self.filters['time_end'] if self.filters['time_end'] is not None else date(2999,12,31)
        time_mask = pd.to_datetime(rank_df['tourney_date']).dt.date.between(time_start, time_end).to_numpy()

        return rank_df.loc[time_mask]
 

    
    def get_yearly_winrate(self):
        '''
        Calculate winrate statistics over time (years)
        
        Returns
        -------
        wr_df: pd.DataFrame
            winrate statistics over time
        '''
        
        wr_df = (self.selected_matches.groupby('year')
                     .agg(matches_played=('winner', np.size),
                          matches_won=('winner', np.sum))
                )

        wr_df['matches_lost'] = wr_df['matches_played'].to_numpy() - wr_df['matches_won'].to_numpy()
        wr_df['win_rate'] = wr_df['matches_won'].to_numpy()/wr_df['matches_played'].to_numpy()
        wr_df.reset_index(inplace=True)
        
        return wr_df
    
    
    def get_surface_winloss(self):
        '''
        Calculate win/loss count by surface
        
        Returns
        -------
        wr_df: pd.DataFrame
            winnloss count by syrface
        '''
        wr_df = self.selected_matches.groupby(['surface', 'result']).size().reset_index()
        wr_df.columns = ['surface', 'result', 'cnt']
        
        return wr_df 
    
         

    def plot_yearly_wr(self):
        
        x = self.yearly_wr['year']
        b1 = self.yearly_wr['matches_won'].to_numpy().astype(int)
        b2 = self.yearly_wr['matches_lost'].to_numpy().astype(int)
        wr = 100*self.yearly_wr['win_rate'].to_numpy().astype(float)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
                go.Bar(
                    x=x, y=b1,
                    name='Matches Won',
                    marker={'color': 'seagreen'},
                    text=b1,
                    textposition='inside',
                    textfont_size=8,
                    opacity=0.8
                ),
            secondary_y=False
        )
        fig.add_trace(
                go.Bar(
                    x=x, y=b2,
                    name='Matches Lost',
                    marker={'color': 'indianred'},
                    text=b2,
                    textposition='inside',
                    textfont_size=8,
                    opacity=0.8
                ),
            secondary_y=False
        )


        fig.add_trace(
                go.Scatter(
                    x=x, y=wr,
                    name='Win Rate',
                    line={'color':'midnightblue', 'width':2},
                    mode='lines+text',
                    text=[str(p) + '%' for i,p in enumerate(np.round(wr, 2))],
                    textposition='top center',
                    textfont_size=8
                ),
             secondary_y=True
        )
        fig.update_layout(
            barmode='stack',
            title={'text': 'Win Rate and Played Matches over Time', 'y':0.9, 'x':0.5,
                   'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'type':'category'},
            yaxis={'range':[0, np.max(b1+b2)+15], 'title': 'Number of Matches'},
            yaxis2={'range':[0, 105], 'title': 'Win Rate (%)'}
        )

        return fig



    def plot_rank(self):
        
        x = self.player_rank['tourney_date'].to_numpy()
        y1 = self.player_rank['rank'].to_numpy()
        y2 = self.player_rank['rank_points'].to_numpy()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=x, y=y1,
                name='Rank',
                marker={'color': 'goldenrod'},
                mode='lines+text',
                text=[p if i%5==0 else None for i, p in enumerate(y1)],
                textposition='top center',
                textfont_size=8,
                opacity=0.8
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=x, y=y2,
                name='Rank Points',
                line={'color':'midnightblue', 'width':2.5},
                mode='lines+text',
                text=[p if i%5==0 else None for i, p in enumerate(y2)],
                textposition='top center',
                textfont_size=8,
                opacity=0.8
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            barmode='stack',
            title={'text': 'Rank and Points over Time', 'y':0.9, 'x':0.5,
                   'xanchor': 'center', 'yanchor': 'top'},
            yaxis={'title': 'Rank', 'range': [0, np.max(y1)+10]},
            yaxis2={'title': 'Rank Points', 'range': [0, np.max(y2)*1.1]},
        )
        
        return fig
    

    def plot_surface_wl(self, surface_colors: Dict = surface_colors):
        
        fig = px.sunburst(self.surface_wl, path=['surface', 'result'], values='cnt', names='cnt')
        fig.data[0].marker.colors = [surface_colors[s.split('/')[0]] for s in fig.data[0].ids]
        
        fig.update_layout(
                    title={'text': 'Win-Loss by Surface', 
                           'y':1, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        )
        
        return fig
    

    def plot_col_overtime(self, col, color='mediumblue'):
        
        m = self.selected_matches
        x = m['tourney_name'] + '(' + m['year'].astype(str) + '), ' + m['round']
        y = m[col]
        txt = m[col].astype(float).round(2).astype(str) + ', ' + m['opponent_name'].apply(get_player_name) + ': ' + m['result']
    
        trace = go.Scatter(
                    x=x, y=y,
                    name='',
                    textposition='top center',
                    hovertemplate=txt,
                    texttemplate=txt,
                    mode='lines+markers',
                    connectgaps=True,
                    marker={'color': color, 'symbol': m['winner']#, 'colorscale': [[0, colors[0]], [1, colors[1]]]
                           },
        ),
    
        layout = go.Layout(
                    height=600,
                    title={'text': f'Percentage {col} over Time', 'y':0.9, 'x':0.5,
                           'xanchor': 'center', 'yanchor': 'top'},
                    xaxis={'title': 'Tournament (Year), Round',
                           'tickangle': 45},
                    yaxis={'title': 'Percentage'}
        )
        
        return go.Figure(data=trace, layout=layout)
    
    def plot_surface_boxplot(self, col: str, surface_colors: Dict = surface_colors):
        
        fig = px.box(self.selected_matches, x=col, color='surface', 
                     color_discrete_map=surface_colors,
                     notched=True
                    )
    
        fig.update_layout(
                    title={'text': f'Summary Statistics of {col} by Surface', 
                           'y':1, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                    xaxis={'title': f'{col} Percentage'},
                    yaxis={'title': 'Surface'},
        )
        return fig
    


    def plot_col_distplot(self, col, colors):
        
        fig = px.histogram(self.selected_matches, x=col, color_discrete_sequence=colors, nbins=50, 
                          marginal='box', histnorm='probability', opacity=0.8)

        fig.update_layout(
                title={'text': f'{col} Distribution', 'y':1, 'x':0.5,
                       'xanchor': 'center', 'yanchor': 'top'},
                xaxis={'title': f'{col}'},
                yaxis={'title': 'Frequency (%)'},
                showlegend=False
        )
        
        return fig
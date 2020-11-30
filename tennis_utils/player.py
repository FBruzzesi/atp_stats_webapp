import pandas as pd
import numpy as np
import dask
from dask import delayed
from statsmodels.stats.proportion import proportion_confint

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
        self.player_matches = (player_matches.loc[player_matches['player_name']==player_name]
                                    .sort_values(['tourney_date', 'match_num'])
        )
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
        self.success_cols = ['ace', 'df', 'firstIn', 'firstWon', 'secondWon', 'returnWon', 'bpConverted', 'bpSaved', 'tbWon', 'decidingSetWon']
        self.total_cols = ['svpt', 'svpt', 'svpt', 'firstIn', 'secondIn', 'returnPlayed', 'bpTotal', 'bpFaced', 'tbPlayed', 'decidingSetPlayed']

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

        self.win_rate = self.selected_matches['result'].value_counts(normalize=True).sort_index()

        self.get_overall_stats()
        self.get_yearly_stats()
        self.get_surface_winloss()
        self.get_h2h()

        self.colors = ['rgb(33,113,181)', 'rgb(217,71,1)', 'rgb(81, 178, 124)', 'rgb(235, 127, 134)'] * 2
            

    def __repr__(self):
        
        return f'{self.player_name}, number of matches: {self.n_matches}'
    

    def select_matches(self):
        '''
        Subsets self.matches based on self.filters value

        '''

        time_start = self.filters['time_start'] if self.filters['time_start'] is not None else date(1970,1,1)
        time_end = self.filters['time_end'] if self.filters['time_end'] is not None else date(2999,12,31)
        time_mask = pd.to_datetime(self.player_matches['tourney_date']).dt.date.between(time_start, time_end).to_numpy()

        masks = [time_mask]
        
        for key, value in self.filters.items():
            if (key not in ['time_start', 'time_end']) and (value is not None) and (value != []):
                m = self.player_matches['opponent_rank'] < value if key=='opponent_rank' else self.player_matches[key].isin(value)
                masks.append(m)

        return self.player_matches.loc[np.all(np.array(masks).T, axis=1)]
        

    def get_rank(self, rank_df):
        '''
        Generate time series of player rank and rank points
        
        Returns
        -------
        rank_df: pd.DataFrame
            rank and rank points over time
        '''
        
        time_start = self.filters['time_start'] if self.filters['time_start'] is not None else date(1970,1,1)
        time_end = self.filters['time_end'] if self.filters['time_end'] is not None else date(2999,12,31)
        time_mask = pd.to_datetime(rank_df['tourney_date']).dt.date.between(time_start, time_end).to_numpy()

        return rank_df.loc[time_mask]
 

    def get_overall_stats(self):

        m = self.selected_matches

        success_overall = pd.Series(m[self.success_cols].fillna(0).to_numpy().sum(axis=0), 
                                    index=[f'success_{c}' for c in self.success_cols]) 
        total_overall = pd.Series(m[self.total_cols].to_numpy().sum(axis=0), 
                                  index=[f'total_{c}' for c in self.success_cols])
        
        self.success_overall = success_overall
        self.total_overall = total_overall
        return self


    def get_yearly_stats(self):
        '''
        Calculate statistics aggregated by year
        '''
        
        m = self.selected_matches

        stats_by_year = (m.groupby('year')
            .agg(
                matches_played=('winner', np.size),
                matches_won=('winner', np.sum),
                ace = ('ace', np.sum),
                df = ('df', np.sum),
                svpt = ('svpt', np.sum),
                firstIn = ('firstIn', np.sum),
                firstWon = ('firstWon', np.sum),
                secondIn = ('secondIn', np.sum),
                secondWon = ('secondWon', np.sum),
                returnWon = ('returnWon', np.sum),    
                returnPlayed = ('returnPlayed', np.sum), 
                bpConverted = ('bpConverted', np.sum),
                bpTotal = ('bpTotal', np.sum),
                bpSaved = ('bpSaved', np.sum),
                bpFaced = ('bpFaced', np.sum),
                tbPlayed = ('tbPlayed', np.sum),
                tbWon = ('tbWon', np.sum),
                decidingSetPlayed = ('decidingSetPlayed', np.sum),
                decidingSetWon = ('decidingSetWon', np.sum))
            .assign(matches_lost = lambda x: x['matches_played'] - x['matches_won'])
            .assign(win_rate = lambda x: x['matches_won']/x['matches_played'])
            .assign(perc_ace = lambda x: x['ace']/x['svpt'])
            .assign(perc_df = lambda x: x['df']/x['svpt'])
            .assign(perc_firstIn = lambda x: x['firstIn']/x['svpt'])
            .assign(perc_firstWon = lambda x: x['firstWon']/x['firstIn'])
            .assign(perc_secondWon = lambda x: x['secondWon']/x['secondIn'])
            .assign(perc_returnWon = lambda x: x['returnWon']/x['returnPlayed'])
            .assign(perc_bpConverted = lambda x: x['bpConverted']/x['bpTotal'])
            .assign(perc_bpSaved = lambda x: x['bpSaved']/x['bpFaced'])
            .assign(tbLost = lambda x: x['tbPlayed'] - x['tbWon'])
            .assign(perc_tbWon = lambda x: x['tbWon']/x['tbPlayed'])
            .assign(decidingSetLost = lambda x: x['decidingSetPlayed'] - x['decidingSetWon'])
            .assign(perc_decidingSetWon = lambda x: x['decidingSetWon']/x['decidingSetPlayed'])
            .reset_index()
        )


        
        lower_df, upper_df = proportion_confint(
            stats_by_year[self.success_cols], 
            stats_by_year[self.total_cols],
            method='wilson'
            )

        lower_df.columns = [f'lower_{c}' for c in self.success_cols]
        upper_df.columns = [f'upper_{c}' for c in self.success_cols]


        self.stats_by_year = stats_by_year
        self.lower_df = lower_df
        self.upper_df = upper_df

        return self
    
    
    def get_surface_winloss(self):
        '''
        Calculate win/loss count by surface
        
        Returns
        -------
        wr_df: pd.DataFrame
            winnloss count by syrface
        '''
        surface_wl = self.selected_matches.groupby(['surface', 'result']).size().reset_index()
        surface_wl.columns = ['surface', 'result', 'cnt']
        
        self.surface_wl = surface_wl
        return self
    

    def get_h2h(self):

        m = self.selected_matches
        h2h = (m.groupby('opponent_name')
            .agg(
                matches_played=('winner', np.size),
                matches_won=('winner', np.sum),
                ace = ('ace', np.sum),
                df = ('df', np.sum),
                svpt = ('svpt', np.sum),
                firstIn = ('firstIn', np.sum),
                firstWon = ('firstWon', np.sum),
                secondIn = ('secondIn', np.sum),
                secondWon = ('secondWon', np.sum),
                returnWon = ('returnWon', np.sum),    
                returnPlayed = ('returnPlayed', np.sum), 
                bpConverted = ('bpConverted', np.sum),
                bpTotal = ('bpTotal', np.sum),
                bpSaved = ('bpSaved', np.sum),
                bpFaced = ('bpFaced', np.sum),
                tbPlayed = ('tbPlayed', np.sum),
                tbWon = ('tbWon', np.sum),
                decidingSetPlayed = ('decidingSetPlayed', np.sum),
                decidingSetWon = ('decidingSetWon', np.sum))
            .assign(matches_lost = lambda x: x['matches_played'] - x['matches_won'])
            .assign(win_rate = lambda x: x['matches_won']/x['matches_played'])
            .assign(perc_ace = lambda x: x['ace']/x['svpt'])
            .assign(perc_df = lambda x: x['df']/x['svpt'])
            .assign(perc_firstIn = lambda x: x['firstIn']/x['svpt'])
            .assign(perc_firstWon = lambda x: x['firstWon']/x['firstIn'])
            .assign(perc_secondWon = lambda x: x['secondWon']/x['secondIn'])
            .assign(perc_returnWon = lambda x: x['returnWon']/x['returnPlayed'])
            .assign(perc_bpConverted = lambda x: x['bpConverted']/x['bpTotal'])
            .assign(perc_bpSaved = lambda x: x['bpSaved']/x['bpFaced'])
            .assign(tbLost = lambda x: x['tbPlayed'] - x['tbWon'])
            .assign(perc_tbWon = lambda x: x['tbWon']/x['tbPlayed'])
            .assign(decidingSetLost = lambda x: x['decidingSetPlayed'] - x['decidingSetWon'])
            .assign(perc_decidingSetWon = lambda x: x['decidingSetWon']/x['decidingSetPlayed'])
            .sort_values('matches_played', ascending=False)
            .reset_index()
        )

        self.h2h = h2h
        return self




    # Plotting Functionalities    

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
    


    def plot_winrate(self):

        fig = go.Figure(
            go.Pie(
                labels=self.win_rate.index,
                values=self.win_rate.to_numpy(),
                marker={'colors': ['indianred', 'seagreen'],
                        'line': {'color':'white', 'width':1}
                }
            )
        )

        fig.update_layout(
            title={'text': 'Overall Win Rate (%)', 'y':0.9, 'x':0.5,
                   'xanchor': 'center', 'yanchor': 'top'},
            legend={'x': .95}

        )
        return fig




    def plot_surface_wl(self, surface_colors: Dict = surface_colors):
        
        fig = px.sunburst(
                data_frame=self.surface_wl, 
                path=['surface', 'result'], 
                values='cnt', 
                names='cnt',
                hover_data=['cnt']
                        )

        fig.data[0].marker.colors = [surface_colors[s.split('/')[0]] for s in fig.data[0].ids]
        
        fig.update_layout(
                    title={'text': 'Win-Loss by Surface', 
                           'y':1, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        )
        
        return fig
    

    def plot_yearly_wr(self):
        
        x = self.stats_by_year['year']
        b1 = self.stats_by_year['matches_won'].to_numpy().astype(int)
        b2 = self.stats_by_year['matches_lost'].to_numpy().astype(int)
        wr = 100*self.stats_by_year['win_rate'].to_numpy().astype(float)

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
            yaxis2={'range':[0, 105], 'title': 'Win Rate (%)'},
            legend={'x': .95}
        )

        return fig



    

    def plot_stats(self,
                   columns
                   ):
     
        success_overall = self.success_overall 
        total_overall = self.total_overall
        stats_by_year = self.stats_by_year 
        upper_df = self.upper_df 
        lower_df = self.lower_df 

        colors = self.colors


        n_cols, n_rows = 2, len(columns)
        specs = [[{}, {'type':'pie'}]] * n_rows
        subplot_titles=[[f'Percentage {c} and 95% CI by year', f'Percentage {c} overall'] for c in columns]


        fig = make_subplots(
                cols=n_cols,
                rows=n_rows,
                specs=specs,
                shared_xaxes=True,
                vertical_spacing=0.05,
                horizontal_spacing=0.05,
                subplot_titles=sum(subplot_titles, []),
                column_widths=[0.8, 0.2]
        )


        x = stats_by_year['year']
        
        for i, col in enumerate(columns):

            fig.add_trace(
                go.Scatter(
                    x=x, y=upper_df[f'upper_{col}'],
                    name=f'{col} upper bound',
                    fill=None, mode='lines', 
                    line=dict(color='darksalmon', width=1)
                ),
                row=i+1, col=1,
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=x, y=lower_df[f'lower_{col}'],
                    name=f'{col} lower bound',
                    fill='tonexty', mode='lines',
                    line=dict(color='darksalmon', width=1)
                ),
                row=i+1, col=1,
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=x, y=stats_by_year[f'perc_{col}'],
                    textposition='top center',
                    name=col,
                    mode='lines+markers',
                    connectgaps=True,
                    marker={'color': colors[i]},
                ),
                row=i+1, col=1,
                secondary_y=False
            )

            fig.add_trace(
                go.Pie(
                    values=[success_overall[f'success_{col}'], total_overall[f'total_{col}'] - success_overall[f'success_{col}']],
                    marker={'colors': ['seagreen', 'indianred'],
                                'line': {'color':'white', 'width':1}}
                ),
                row=i+1, col=2
            )

            
        # Layout
        fig.update_layout({
            'showlegend': False,
            'xaxis': {'title': 'Year'},
            'yaxis': {'title': 'Percentage'},
            **{f'xaxis{i}': {'title': 'Year'} for i in range(1, n_rows+1)},
            **{f'yaxis{i}': {'title': 'Percentage'} for i in range(1, n_rows+1)},
        })

        return fig





    def plot_distribution(self,
                        columns, 
                        ):


        m = self.selected_matches
        colors = self.colors

        n_cols, n_rows = 2, len(columns)
        specs = [[{}, {}]] * n_rows
        subplot_titles=[[f'{c} Boxplot', f'{c} Distplot'] for c in columns]
        
        fig = make_subplots(
                cols=n_cols,
                rows=n_rows,
                specs=specs,
                shared_xaxes=True,
                shared_yaxes=True,
                vertical_spacing=0.08,
                horizontal_spacing=0.02,
                subplot_titles=sum(subplot_titles, []),
                column_widths=[0.8, 0.2]
        )

        

        for i, col in enumerate(columns):
            

            fig.add_trace(
                go.Box(
                    x=m['year'],
                    y=m[f'perc_{col}'],
                    marker_color=colors[i]
                ),
                row=i+1, col=1  
            )


            hist, kde = ff.create_distplot([m[f'perc_{col}'].to_numpy()], bin_size=0.015,
                            group_labels=[col], show_rug=False, colors=[colors[i]],
                            histnorm='probability',
                        )['data']
            
            hist_ = go.Histogram(
                y=hist['x'],
                histnorm='probability',
                ybins = {'start':hist['xbins']['start'], 'end':hist['xbins']['end'], 'size':hist['xbins']['size']},
                opacity=0.7, marker_color=colors[i]
            )

            kde_ = go.Scatter(
                x=kde['y'],
                y=kde['x'],
                marker_color=colors[i]
            )

            fig.add_trace(hist_,
                row=i+1, col=2
            )

            fig.add_trace(
                kde_,
                row=i+1, col=2
            )



        # Layout
        fig.update_layout({
            'showlegend': False,
            f'xaxis{n_rows*n_cols-1}': {'title': 'Year'},
            f'xaxis{n_rows*n_cols}': {'title': 'Frequency'},
            **{f'yaxis{2*r-1}': {'title': 'Percentage'} for r in range(1, n_rows+1)}
        })

        return fig


    def plot_h2h(self):

        h2h = self.h2h.iloc[:15]

        x = h2h['opponent_name']
        b1 = h2h['matches_won']
        b2 = h2h['matches_played'] - h2h['matches_won']
        wr = 100*h2h['win_rate']

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
                mode='markers+text',
                text=[str(p) + '%' for i,p in enumerate(np.round(wr, 2))],
                textposition='top center',
                textfont_size=8
            ),
             secondary_y=True
        )
        
        fig.update_layout(
            barmode='stack',
            title={'text': 'Win Rate with most played opponents', 'y':0.9, 'x':0.5,
                   'xanchor': 'center', 'yanchor': 'top'},
            xaxis={'type':'category', 'title': 'Opponent name'},
            yaxis={'range':[0, np.max(b1+b2)+15], 'title': 'Number of Matches'},
            yaxis2={'range':[0, 110], 'title': 'Win Rate (%)'},
        )

        return fig
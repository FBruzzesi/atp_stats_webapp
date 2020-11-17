#import random
from collections import Counter
from typing import List, Set, Tuple, Dict
from datetime import date
import numpy as np
import pandas as pd

from datetime import date, datetime
import os, re
from plotly.offline import init_notebook_mode
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings("ignore")

opp_name = lambda n: '. '.join(['.'.join([e[0] for e in n.split(' ')[:-1]]),  n.split(' ')[-1]])


class TennisPlayer:
    '''
    Attributes
    ----------
    player_name: str
        Name of the tennis player
    time_start: datetime.date
        Start of the period to consider
    time_end: datetime.date
        End of the period to consider
    all_matches: pd.DataFrame
        All matches the player played during the time period
    tournaments: List[str]
        List of tournaments to consider
    surface: List[str]
        List of surfaces to consider
    opponents: List[str]
        List of opponents to consider   
    matches: pd.DataFrame
        Matches filtered for tournaments, surface and opponents
    n_matches: int
        Lenght of matches, hence number of matches played with these filters


    Methods
    -------
    '''
    def __init__(self, 
                 player_name: str, 
                 raw_matches: pd.DataFrame, 
                 pdata: pd.DataFrame,
                 time_start: date = None,
                 time_end: date = None,
                 tournaments: List[str] = None,
                 surface: List[str] = None,
                 opponents: List[str] = None,
                 init_stats: bool = True):
        
        assert isinstance(raw_matches, pd.DataFrame)
        assert isinstance(player_name, str), 'Must select a single player'
        
        self.player_name = player_name
        self.time_start = time_start if time_start is not None else date(1970,1,1)
        self.time_end = time_end if time_end is not None else date(2999,12,31)

        self.tournaments = tournaments
        self.surface = surface
        self.opponents = opponents
        
        self.matches, self.rnk_df, self.player_details = self.init_player(raw_matches, pdata)

        self.n_matches = self.matches.shape[0]
        
        if init_stats:    
            self.yearly_wr = self.get_yearly_winrate()
            self.surface_wl = self.get_surface_winloss()
            
    def __repr__(self):
        
        return f'{self.player_name}, number of matches: {self.n_matches}'
    
    
    def init_player(self, raw_matches, pdata):
        '''
        Initialize matches and rank dataframe
        
        Returns
        -------
        matches_df: pd.DataFrame
            containes matches with applied filters
        rnk_df: pd.DataFrame
            rank and rank points over time
        '''
        
        
        player_mask = raw_matches['name'].eq(self.player_name).to_numpy()
        time_mask = pd.to_datetime(raw_matches['tourney_date']).dt.date.between(self.time_start, self.time_end).to_numpy()
            
        masks =  np.array([player_mask, time_mask]).T
        raw_matches = raw_matches.loc[np.all(masks, axis=1)]

        rnk_df = raw_matches[['tourney_date', 'rank', 'rank_points']].drop_duplicates().sort_values('tourney_date').dropna(subset=['rank', 'rank_points'])
        rnk_df[['rank', 'rank_points']] = rnk_df[['rank', 'rank_points']].astype(int)

        m, _ = raw_matches.shape

        tourney_mask = raw_matches['tourney_name'].isin(self.tournaments).to_numpy() if self.tournaments is not None else np.ones(m)
        surface_mask = raw_matches['surface'].isin(self.surface).to_numpy() if self.surface is not None else np.ones(m)
        opponents_mask = raw_matches['opponent_name'].isin(self.opponents).to_numpy() if self.opponents is not None else np.ones(m)

        masks = np.array([tourney_mask, surface_mask, opponents_mask]).T
        matches_df = raw_matches.loc[np.all(masks, axis=1)]

        matches_df.reset_index(inplace=True, drop=True)
        
        pdata_ = pdata[pdata['id'] == matches_df['id'].max()].reset_index(drop=True).copy()
        pdata_['birth_date'] = pdata_['birth_date'].astype('datetime64[ns]')
        
        pdetails = pd.DataFrame(data={
                        'Field': ['Firstname', 'Lastname', 'Birthdate', 'Age', 'Nationality', 'Hand', 'Height', 'Best Rank'],
                        'Value': [pdata_.loc[0, 'first_name'], pdata_.loc[0, 'last_name'], str(pdata_.loc[0, 'birth_date'])[:10], 
                                  pdata_.loc[0, 'age'],  pdata_.loc[0, 'country_code'], pdata_.loc[0,'hand'], 
                                  matches_df['ht'].max(skipna=False),  int(matches_df['rank'].min())]
                    })
        
        return matches_df, rnk_df, pdetails
    
       
    
    
    def get_yearly_winrate(self):
        '''
        Calculate winrate statistics over time (years)
        
        Returns
        -------
        wr_df: pd.DataFrame
            winrate statistics over time
        '''
        
        wr_df = (self.matches.groupby('year')
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
        wr_df = self.matches.groupby(['surface', 'result']).size().reset_index()
        wr_df.columns = ['surface', 'result', 'cnt']
        
        return wr_df 
    

    
    
    
    
    
    
    
class RenderTennisPlayer(TennisPlayer):
    
    def __init__(self, 
                 player_name:str,  
                 raw_matches:pd.DataFrame, 
                 pdata:pd.DataFrame,
                 time_start: date = None,
                 time_end: date = None,
                 tournaments: List[str] = None,
                 surface: List[str] = None,
                 opponents: List[str] = None,
                 init_stats:bool = True):
        
        super().__init__(player_name,
                         raw_matches,
                         pdata,
                         time_start,
                         time_end,
                         tournaments,
                         surface,
                         opponents,
                         init_stats)
    
    
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
        
        x = self.rnk_df['tourney_date'].to_numpy()
        y1 = self.rnk_df['rank'].to_numpy()
        y2 = self.rnk_df['rank_points'].to_numpy()
        
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
            yaxis2={'title': 'Rank Points', 'range': [0, np.max(y2)*1.1]}
        )
        
        return fig
    
    def plot_surface_wl(self, surface_colors={'Clay':'firebrick', 'Grass':'seagreen', 'Hard':'midnightblue', 'Carpet':'limegreen'}):
        
        fig = px.sunburst(self.surface_wl, path=['surface', 'result'], values='cnt', names='cnt')
        fig.data[0].marker.colors = [surface_colors[s.split('/')[0]] for s in fig.data[0].ids]
        
        fig.update_layout(
                    title={'text': f'Win-Loss by Surface', 
                           'y':1, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        )
        
        return fig
    
    def plot_col_overtime(self, col, color='mediumblue'):
        
        m = self.matches
        x = m['tourney_name'] + '(' + m['year'].astype(str) + '), ' + m['round']
        y = m[col]
        txt = m[col].astype(float).round(2).astype(str) + ', ' + m['opponent_name'].apply(opp_name) + ': ' + m['result']
    
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
    
    def plot_surface_boxplot(self, 
                             col:str, 
                             surface_colors: Dict = {'Clay':'firebrick', 
                                                     'Grass':'seagreen', 
                                                     'Hard':'midnightblue', 
                                                     'Carpet':'limegreen'}
                            ):
        
        assert col in self.matches.columns
        
        fig = px.box(self.matches, x=col, color='surface', 
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
        
        fig = px.histogram(self.matches, x=col, color_discrete_sequence=colors, nbins=50, 
                          marginal='box', histnorm='probability', opacity=0.8)

        fig.update_layout(
                title={'text': f'{col} Distribution', 'y':1, 'x':0.5,
                       'xanchor': 'center', 'yanchor': 'top'},
                xaxis={'title': f'{col}'},
                yaxis={'title': 'Frequency (%)'},
                showlegend=False
        )
        
        return fig
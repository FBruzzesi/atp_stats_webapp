import pandas as pd
import numpy as np
import dask
from dask import delayed
import os, warnings, multiprocessing as mp

from datetime import datetime as dt
from tennis_utils.settings import scraper_settings

t_cols = scraper_settings['t_cols']
str_cols = scraper_settings['str_cols']
int_cols = scraper_settings['int_cols']
float_cols = scraper_settings['float_cols']
fill_values = scraper_settings['fill_values']
w_cols = scraper_settings['w_cols']
l_cols = scraper_settings['l_cols']
wl_cols = scraper_settings['wl_cols']


warnings.filterwarnings("ignore")


n_cpu = mp.cpu_count()
dask.config.set(pool=mp.pool.ThreadPool(n_cpu))


def timer(f, *args, **kwargs):
    '''
    timer decorator
    '''
    def wrapper(*args, **kwargs):
        tic = dt.now()
        result = f(*args, **kwargs)
        toc = dt.now()
        print(f'@{f.__name__} took {toc-tic}')
        return result
    return wrapper
    


class SackmanScraper:
    '''
    Simple scraper for JeffSackmann tennis atp github repo
    Attributes
    ----------
    main_path: str
        path pointin to JeffSackmann tennis atp repository
    players: pd.DataFrame
        dataframe containing players details
    matches: pd.DataFrame
        dataframe containing matches between players
    tournaments: pd.DataFrame
        dataframe containing tournaments details
    prepared_matches: bool
        boolean value to determine whether or not matches dataframe 
        has been preprocessed
    wl_matches: pd.DataFrame
        dataframe containing matches data for each player i.e.
        each matches row becomes two in wl_matches

    Methods
    -------
    get_players
        retrieves tennis players details
    get_matches
        retrieves tennis matches data
    prepare_matches
        preprocesses matches data
    split_wl_matches
        splits matches into win/loss matches
    save_players
        dumps player details data
    save_matches
        dumps matches data 
    save_tournaments
        dumps tournaments data
    save_wl_matches
        dumps wl matches data
    '''
    def __init__(self):
        
        self.main_path = 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/'
        self.players = None
        self.matches = None
        self.tournaments = None 
        self.prepared_matches = False
        self.wl_matches = None
        
              
    def __repr__(self):
         return f'Scraper of {self.main_path}'

    @timer
    def get_players(self):
        '''
        Retrieves player details from https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv 
        and store them in self.players as a pd.DataFrame
        '''
        players_path = self.main_path + 'atp_players.csv'
        player_cols = ['id', 'first_name', 'last_name', 'hand', 'birth_date', 'country_code']
        self.players = pd.read_csv(players_path, names=player_cols)

        self.players = self.players[self.players['birth_date']>19500101]
        self.players = (self.players.assign(birth_date = pd.to_datetime(self.players['birth_date'], format='%Y%m%d').dt.date)
                                    .assign(player_name = self.players['first_name'] + ' ' + self.players['last_name'])
        )
        self.players.rename(columns={'birth_date': 'birthdate'}, inplace=True)
        return self


    @timer
    def save_players(self, destination_path, type='parquet', sep=','):
        '''
        Save self.players pd.DataFrame into destination_path

        Parameters
        ----------
        destination_path: str or Path
        type: str, default 'parquet' 
            Type for saving the dataframe self.players
            One between 'parquet' or 'csv'. 
        sep: str, default ','
            Field delimiter for the output file if type='csv'
        '''
        
        if not isinstance(self.players, pd.DataFrame):
            print('Players data is not initialized yet, retrieving it now...')

            self.get_players()

        if type == 'parquet':
            self.players.to_parquet(destination_path, engine='pyarrow', index=False)
            print(f'Saved as parquet file in {destination_path}')
        elif type == 'csv':
            self.players.to_csv(destination_path, index=False, sep=sep)
            print(f'Saved as csv file in {destination_path}')
        else: raise Exception('Can only save in csv and parquet format')


    @timer
    def get_matches(self, years_to_parse=list(range(2000, 2021)), drop_davis=False):
        '''
        Retrieves matches data from https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/ 
        Stores matches data in self.matches
        Stores tournament data in self.tournaments

        Parameters
        ----------
        years_to_parse: List, default [2000, 2001, ..., 2019, 2020]
            List of years to retrieve matches for
        drop_davis: bool, default False
            Either to drop or keep davis matches
        '''
        dfs = []

        for yr in years_to_parse:
            for sfx in ['', 'futures_', 'qual_chall_']:
            
                matches_path = self.main_path + f'atp_matches_{sfx}{yr}.csv'
                tmp_df = delayed(pd.read_csv)(matches_path)
                dfs.append(tmp_df)
        
        result = delayed(pd.concat)(dfs)

        self.matches = result.compute()
        self.columns = self.matches.columns
        
        if drop_davis:
            self.matches = self.matches[~self.matches['tourney_name'].str.startswith('Davis Cup')]

        self.tournaments = self.matches[t_cols].copy().drop_duplicates()
        self.tournaments[t_cols] = self.tournaments[t_cols].astype(str)

        return self
        

    @timer
    def prepare_matches(self):
        '''
        Preprocesses self.matches data
        '''

        if not isinstance(self.matches, pd.DataFrame):
            print('Matches data is not initialized yet, retrieving it now...')

            self.get_matches()
        
        m = self.matches

        

        with pd.option_context('mode.use_inf_as_na', True):
             m = m.fillna(value=fill_values)

        m[str_cols] = m[str_cols].astype(str)
        m[int_cols] = m[int_cols].astype(int)
        m[float_cols] = m[float_cols].astype(float)

        masks = np.array([
                    ~m['minutes'].isna(),
                    m['w_svpt']!=0,
                    m['l_svpt']!=0,
                    ~m['surface'].isna(),
        ])
        m = m[np.all(masks.T, axis=1)]
                
        self.matches = (m
                .assign(w_percAce = m['w_ace']/m['w_svpt'])
                .assign(l_percAce = m['l_ace']/m['l_svpt'])
                .assign(w_percDf = m['w_df']/m['w_svpt'])
                .assign(l_percDf = m['l_df']/m['l_svpt'])
                .assign(w_perc1stIn = m['w_1stIn'] / m['w_svpt'])
                .assign(l_perc1stIn = m['l_1stIn'] / m['l_svpt'])
                .assign(w_perc1stWon = m['w_1stWon'] / m['w_1stIn'])
                .assign(l_perc1stWon = m['l_1stWon'] / m['l_1stIn'])
                .assign(w_perc2ndWon = m['w_2ndWon'] / (m['w_svpt'] - m['w_1stIn']))
                .assign(l_perc2ndWon = m['l_2ndWon'] / (m['l_svpt'] - m['l_1stIn']))
                .assign(w_percBpSave = np.where(m['w_bpFaced']==0, 1, m['w_bpSaved'] / m['w_bpFaced']))
                .assign(l_percBpSave = np.where(m['l_bpFaced']==0, 1, m['l_bpSaved'] / m['l_bpFaced']))
                .assign(w_SvLost = m['w_bpFaced'] - m['w_bpSaved'])
                .assign(l_SvLost = m['l_bpFaced'] - m['l_bpSaved'])
                .assign(w_bpWon = lambda x: x['l_SvLost'].to_numpy())
                .assign(l_bpWon = lambda x: x['w_SvLost'].to_numpy())
                .assign(w_percBpWon = lambda x: np.where(x['l_bpFaced']!=0, x['w_bpWon']/x['l_bpFaced'], np.nan))
                .assign(l_percBpWon = lambda x: np.where(x['w_bpFaced']!=0, x['l_bpWon']/x['w_bpFaced'], np.nan))
                .assign(w_percSvLost = lambda x: x['w_SvLost'] / x['w_SvGms'])
                .assign(l_percSvLost = lambda x: x['l_SvLost'] / x['l_SvGms'])
                .assign(w_returnWon = m['l_svpt'] - m[['l_1stWon', 'l_2ndWon']].sum(axis=1))
                .assign(l_returnWon = m['w_svpt'] - m[['w_1stWon', 'w_2ndWon']].sum(axis=1))
                .assign(w_percReturnWon = lambda x: x['w_returnWon']/x['l_svpt'])
                .assign(l_percReturnWon = lambda x: x['l_returnWon']/x['w_svpt'])
                .assign(w_percServePointsWon = m[['w_1stWon', 'w_2ndWon']].sum(axis=1)/m['w_svpt'])
                .assign(l_percServePointsWon = m[['l_1stWon', 'l_2ndWon']].sum(axis=1)/m['l_svpt'])
                )

        self.prepared_matches = True 
        return self
    

    @timer
    def save_matches(self, destination_path, type='parquet', sep=','):
        '''
        Save self.matches pd.DataFrame into destination_path

        Parameters
        ----------
        destination_path: str or Path
        type: str, default 'parquet' 
            Type for saving the dataframe self.players
            One between 'parquet' or 'csv'. 
        sep: str, default ','
            Field delimiter for the output file if type='csv'
        '''
        
        if not self.prepared_matches:
            print('Matches data is not prepared, preparing it now...')

            self.prepare_matches()

        print('Saving matches data...')
        if type == 'parquet':
            self.matches.to_parquet(destination_path, engine='pyarrow', index=False)
            print(f'Saved as parquet file in {destination_path}')
        elif type == 'csv':
            self.matches.to_csv(destination_path, index=False, sep=sep)
            print(f'Saved as csv file in {destination_path}')
        else: raise Exception('Can only save in csv and parquet format')


    @timer
    def save_tournaments(self, destination_path, type='parquet', sep=','):
        '''
        Save self.tournaments dataframe into destination_path

        Parameters
        ----------
        destination_path: str or Path
        type: str, default 'parquet' 
            Type for saving the dataframe self.players
            One between 'parquet' or 'csv'. 
        sep: str, default ','
            Field delimiter for the output file if type='csv'
        '''
        
        if not isinstance(self.tournaments, pd.DataFrame):
            print('Tournaments data is not initialized, retrieving it now...')

            self.get_matches()

        print('Saving tournaments data...')
        if type == 'parquet':
            self.tournaments.to_parquet(destination_path, engine='pyarrow', index=False)
            print(f'Saved as parquet file in {destination_path}')
        elif type == 'csv':
            self.tournaments.to_csv(destination_path, index=False, sep=sep)
            print(f'Saved as csv file in {destination_path}')
        else: raise Exception('Can only save in csv and parquet format')



    @timer 
    def split_wl_matches(self):
        '''
        Splits matches dataframe into winner and loser 
        to create dataframe based on each tennis player match
        '''

        if not isinstance(self.matches, pd.DataFrame) or not self.prepared_matches:
            print('Matches data is not prepared, preparing it now...')

            self.prepare_matches()               
        
        w_matches = self.matches[w_cols]
        w_matches.columns = wl_cols
        w_matches = (w_matches.assign(winner = 1)
                              .assign(result = 'Won')
        )

        l_matches = self.matches[l_cols]
        l_matches.columns = wl_cols
        l_matches = (l_matches.assign(winner = 0)
                              .assign(result = 'Lost')
        )

        self.wl_matches = (pd.concat([w_matches, l_matches])
                             .sort_values(['tourney_id', 'match_num', 'winner'])
        )

        self.wl_matches = (self.wl_matches.assign(year = pd.to_datetime(self.wl_matches['tourney_date'], format='%Y%m%d').dt.year)
                                          .assign(tourney_date = pd.to_datetime(self.wl_matches['tourney_date'], format='%Y%m%d').dt.date)
        )

        return self


    @timer
    def save_wl_matches(self, destination_path, type='parquet', sep=','):
        '''
        Save self.wl_matches dataframe into destination_path

        Parameters
        ----------
        destination_path: str or Path
        type: str, default 'parquet' 
            Type for saving the dataframe self.players
            One between 'parquet' or 'csv'. 
        sep: str, default ','
            Field delimiter for the output file if type='csv'
        '''
        
        if not isinstance(self.wl_matches, pd.DataFrame):
            print('Win-Loss matches data is not initialized, retrieving it now...')

            self.split_wl_matches()

        print('Saving tournaments data...')
        if type == 'parquet':
            self.wl_matches.to_parquet(destination_path, engine='pyarrow', index=False)
            print(f'Saved as parquet file in {destination_path}')
        elif type == 'csv':
            self.wl_matches.to_csv(destination_path, index=False, sep=sep)
            print(f'Saved as csv file in {destination_path}')
        else: raise Exception('Can only save in csv and parquet format')

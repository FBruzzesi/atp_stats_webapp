from tennis_utils.scrapers import SackmanScraper
from tennis_utils.player import TennisPlayer, TennisDataLoader
import pandas as  pd
import os
    

data_path = os.getcwd() + '/data'

# tdl = TennisDataLoader(data_path + '/matches.parquet', data_path + '/players.parquet')
# print(tdl.matches['round'].unique())


s = SackmanScraper()
print(s)

s.save_players(data_path + '/players.parquet')
s.save_matches(data_path + '/raw_matches.parquet')
s.save_tournaments(data_path + '/tournaments.parquet')
s.save_wl_matches(data_path + '/matches.parquet')

# # %%
# 
# tdata = pd.read_parquet(destination_path + '/tournaments.parquet')
# pdata = pd.read_parquet(destination_path + '/players_data.parquet')
# # %%
# print(mdata[['tourney_date', 'year']].head())
# #%%

# rf_matches = mdata[mdata['name'] == 'Roger Federer']
# rf_det = pdata[pdata['player_name'] == 'Roger Federer']

# tp = RenderTennisPlayer('Roger Federer', rf_matches, rf_det, opponents=['Rafael Nadal'])
# print(tp.matches.shape, tp.selected_matches.shape, tp.matches['year'].unique())

# %%

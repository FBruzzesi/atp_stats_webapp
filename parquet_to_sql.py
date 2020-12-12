from tennis_utils.player import TennisDataLoader
import pandas as  pd
import os
    
import sqlite3 

data_path = os.getcwd() + '/data'

tdl = TennisDataLoader(data_path + '/matches.parquet', data_path + '/players.parquet')
matches_df, players_df = tdl.matches, tdl.players

conn = sqlite3.connect(data_path + '/tennis_database.db')
c = conn.cursor()

matches_df.to_sql('matches', conn, if_exists='replace')
players_df.to_sql('players', conn, if_exists='replace')

matches_df = pd.read_sql("SELECT * from matches", conn)
players_df = pd.read_sql("SELECT * from players", conn)

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',100)

            
#%%  

# import random


# class Game(object):
    
#     def __init__(self, p:float):
        
#         self.p = p
#         self.s: int = 0
#         self.r: int = 0
#         self.done: bool= False
#         self.result: list = [0,0]
        
#     def simulate(self) -> int:
        
        
#         while not self.done:
            
#             if random.uniform(0,1) <= self.p:
#                 self.s += 1
#             else: self.r += 1
            
#             if self.s >= 4 and self.s - self.r >= 2:
#                 self.result[0] += 1
#                 self.done = True
                
#             elif self.r >= 4 and self.r - self.s >= 2:
#                 self.result[1] += 1
#                 self.done = True
                
#         self.game_score = [self.s, self.r]
#         print(self.game_score)
        
        
#         return np.argmax(self.result)
    

# for _ in range(12):
#     G = Game(0.6)
#     print(G.simulate())

#%%

from numba import njit
import random

import time                                                

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print(f'Method {method.__name__} took: {te-ts} seconds')
        return result

    return timed


@njit() 
def simulate_game(p: float) -> int:
    
    s: int = 0
    r: int = 0
    done: bool = False
    result: int = 0

    while not done:

        if random.uniform(0,1) <= p:
            s += 1
        else: r += 1
        
        if s>=4 and s-r >= 2:
            result = 0
            done = True
            
        elif r>=4 and r-s>=2:
            result = 1
            done = True
    
    print('Game Score: ', s, r)
    
    return result
    


class Game:

    def __init__(self, p):
        
        self.p = p
        self.s = 0
        self.r = 0
        self.done = False
        self.score = [0,0]
    
    @timeit
    def simulate(self):
        return simulate_game(self.p)
    
    
    
    

    
class TieBreak:
    
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        
    @timeit
    def simulate(self):
        return simulate_tiebreak(self.p1, self.p2)
    
    

G = Game(0.6)
T = TieBreak(0.6, 0.65)

for _ in range(10):
    
    print(T.simulate())


#%%
    
class TennisPlayer:
    
    def __init__(self, 
                 name: str,
                 prob_serve_1st: np.ndarray, 
                 prob_win_1st: np.ndarray,
                 prob_win_2nd: np.ndarray,
                 prob_win_return : np.ndarray,
                 break_point_save_rate: np.ndarray,
                 break_point_conversion_rate: np.ndarray,
                 rank: np.ndarray, 
                 dt: np.ndarray,
                 opponent: np.ndarray
                 ) -> None :
      
        '''
        Parameters
        ----------
        name : str
            Player Name
            
        prob_serve_1st : np.ndarray
            Probability of serving 1st ball
        
        prob_win_1st : np.ndarray
            Probability of winning the point on 1st serve
        
        prob_win_2nd : np.ndarray
             Probability of winning the point on 2nd serve
        
        prob_win_return : np.ndarray
            Probability of winning the point on return
        
        break_point_save_rate : np.ndarray
            Probability of winning the point on serving break point 
            
        break_point_conversion_rate : np.ndarray
            Probability of winning the point on returning break point 

        rank: np.ndarray
            Player rank
            
        dt: np.ndarray
            Match statistic date
        
        opponent: np.ndarray
            Opponent Name
            
        '''
        
        
        if not isinstance(name, str): raise TypeError('name has to be a string')
        
        if not isinstance(prob_serve_1st, np.ndarray): raise TypeError('prob_serve_1st has to be a np.ndarray')
        if not isinstance(prob_win_1st, np.ndarray): raise TypeError('prob_win_1st has to be a np.ndarray')
        if not isinstance(prob_win_2nd, np.ndarray): raise TypeError('prob_win_2nd has to be a np.ndarray')
        if not isinstance(prob_win_return, np.ndarray): raise TypeError('prob_win_return has to be a np.ndarray')
        if not isinstance(break_point_save_rate, np.ndarray): raise TypeError('break_point_save_rate has to be a np.ndarray')
        if not isinstance(break_point_conversion_rate, np.ndarray): raise TypeError('break_point_conversion_rate has to be a np.ndarray')
        if not isinstance(rank, np.ndarray): raise TypeError('rank has to be a np.ndarray')
        if not isinstance(dt, np.ndarray): raise TypeError('dt has to be a np.ndarray')
        if not isinstance(opponent, np.ndarray): raise TypeError('opponent has to be a np.ndarray')

        if not prob_serve_1st.ndim == 1: raise Exception('prob_serve_1st has to be a 1-dimensional')
        if not prob_win_1st.ndim == 1: raise Exception('prob_win_1st has to be a 1-dimensional')
        if not prob_win_2nd.ndim == 1: raise Exception('prob_win_2nd has to be a 1-dimensional')
        if not prob_win_return.ndim == 1: raise Exception('prob_win_return has to be a 1-dimensional')
        if not break_point_save_rate.ndim == 1: raise Exception('break_point_save_rate has to be a 1-dimensional')
        if not break_point_conversion_rate.ndim == 1: raise Exception('break_point_conversion_rate has to be a 1-dimensional')
        if not rank.ndim == 1: raise Exception('rank has to be a 1-dimensional')
        if not dt.ndim == 1: raise Exception('dt has to be a 1-dimensional')
        if not opponent.ndim == 1: raise Exception('opponent has to be a 1-dimensional')

        if not np.logical_and(np.all(prob_serve_1st <= 1), np.all(prob_serve_1st >= 0)): raise Exception('prob_serve_1st values must be in (0,1) range')
        if not np.logical_and(np.all(prob_win_1st <= 1), np.all(prob_win_1st >= 0)): raise Exception('prob_win_1st values must be in (0,1) range')
        if not np.logical_and(np.all(prob_win_2nd <= 1), np.all(prob_win_2nd >= 0)): raise Exception('prob_win_2nd values must be in (0,1) range')
        if not np.logical_and(np.all(prob_win_return <= 1), np.all(prob_win_return >= 0)): raise Exception('prob_win_return values must be in (0,1) range')
        if not np.logical_and(np.all(break_point_save_rate <= 1), np.all(break_point_save_rate >= 0)): raise Exception('break_point_save_rate values must be in (0,1) range')
        if not np.logical_and(np.all(break_point_conversion_rate <= 1), np.all(break_point_conversion_rate >= 0)): raise Exception('break_point_conversion_rate values must be in (0,1) range')
        if not np.logical_and(np.all(rank > 0), rank.dtype == int): raise Exception('rank values must be positive integers')

        if not prob_serve_1st.size == prob_win_1st.size == prob_win_2nd.size == prob_win_return.size == break_point_save_rate.size == break_point_conversion_rate.size == rank.size == dt.size == opponent.size:
            raise Exception('Parameters has to be all the same size')
        
        
        self.name = name
        self.ps1 = prob_serve_1st
        self.pw1 = prob_win_1st
        self.pw2 = prob_win_2nd
        self.pwr = prob_win_return
        self.bpsr = break_point_save_rate
        self.bpcr = break_point_conversion_rate
        self.rank = rank
        self.dt = dt
        self.opponent = opponent
        
    def __str__(self):
        
        return f'Tennis Player Instance: {self.name}'
    
    def __repr__(self):
        
        return f'Tennis Player Instance: {self.name}'
    
    
        
        
#%%
n = 'Roger'

ps1 = np.random.rand(10)
pw1 = np.random.rand(10)
pw2 = np.random.rand(10)
pwr = np.random.rand(10)

bpsr = np.random.rand(10)
bpcr = np.random.rand(10)

rank = np.random.randint(1,5,10)
dt = np.random.rand(10)
op = np.random.rand(10)



rf = TennisPlayer('Roger Federer', ps1, pw1, pw2, pwr, bpsr, bpcr, rank, dt, op)
rn = TennisPlayer('Rafael Nadal', ps1, pw1, pw2, pwr, bpsr, bpcr, rank, dt, op)



from numba import jit

@njit 
def simulate_game(p: float) -> int:
    
    s: int = 0
    r: int = 0
    game_done: bool = False
    game_result: int = 0

    while not game_done:

        if random.uniform(0,1) <= p:
            s += 1
        else: r += 1
        
        if s>=4 and s-r >= 2:
            game_result = 0
            game_done = True
            
        elif r>=4 and r-s>=2:
            game_result = 1
            game_done = True
        
    return game_result


#@jit() 
def simulate_tiebreak(p1: float, p2: float) -> int:
    
    s1: int = 0
    s2: int = 0
    tb_done: bool = False
    tb_result: int = 0
    tb_turn    :int = 0
    player_to_serve: list = [p1, p2, p2, p1]
    
    p = player_to_serve[tb_turn % 4]
    while not tb_done:
        
        if tb_turn % 4 in [0,3]:
            if random.uniform(0,1) <= p:
                s1 += 1
            else: s2 += 1
        elif tb_turn % 4 in [1,2]:
            if random.uniform(0,1) <= p:
                s2 += 1
            else: s1 += 1
        
        tb_turn += 1
        p = player_to_serve[tb_turn % 4]
        if s1>=7 and s1-s2 >= 2:
            tb_result = 0
            tb_done = True
            
        elif s2>=7 and s2-s1>=2:
            tb_result = 1
            tb_done = True
    
    print('Tiebreak Score: ', s1, s2)
    
    return tb_result

#@jit()
def simulate_set(p1, p2, turn_start):
    
    set_score = [0, 0]
    players = [p1, p2]
    set_done = False
    
    serve_turn = turn_start
    player_to_serve = players[serve_turn]
    while not set_done:
        
        
        if set_score != [6,6]:
            game_result = simulate_game(player_to_serve)

            if game_result == 0:
                set_score[serve_turn] +=1
            else: set_score[1-serve_turn] +=1
            
            serve_turn = (serve_turn+1) % 2
            player_to_serve = players[serve_turn]
            
            if set_score[0] >= 6 and set_score[0]-set_score[1]>=2:
                set_done = True
            if set_score[1] >= 6 and set_score[1]-set_score[0]>=2:
                set_done = True
        
        else:
            tb_result = simulate_tiebreak(*players) if turn_start == 0 else simulate_tiebreak(*players[::-1])
            
            if tb_result == turn_start == 0 or  tb_result == turn_start == 1:
                set_score = [7, 6]
            else:
                set_score = [6, 7]
            
            set_done = True

    return set_score        
        
#@jit()
def simulate_match(p1, p2, is_slam=False):
    
    
    n_set: int = 5 if is_slam else 3
    
    # players = [p1, p2]
    l: list = [0, 1]
    serve_turn:int = random.choice(l)
    match_score = [ [0, 0] for _ in range(n_set)]
    
    match_done = False
    s = 0
    while not match_done:
        
        set_score = simulate_set(p1, p2, turn_start=serve_turn)
        match_score[s] = set_score
        
        
        if (set_score[0] + set_score[1])%2 == 1:
            serve_turn = 1 - serve_turn 
            
        s+=1
        if s == n_set:
            match_done=True
        
    return match_score
        
        
    
    
    
    
    
    
    
class Match:
    
    def __init__(self, player1, player2):
        
        '''
        Parameters
        ----------
        player1 : Tennis Player Instace
            
        player2 : Tennis Player Instace        
            
        '''
        
        if not isinstance(player1, TennisPlayer): raise TypeError('Player1 has to be a TennisPlayer instance')
        if not isinstance(player2, TennisPlayer): raise TypeError('Player2 has to be a TennisPlayer instance')

        self.p1 = player1
        self.p2 = player2
        
        
    def simulate(self, n=1000):
        
        return simulate_match(n)
        
    

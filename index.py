import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# Local Imports
from app import app, server
import callbacks

from utils.filter_rows import get_filter_rows


tab_style={'fontWeight': 'bold'}

# # Load data
# data_path = os.getcwd() + '/data'

# tdl = TennisDataLoader(data_path=data_path)
matches_df, players_df = callbacks.matches_df, callbacks.players_df


header=html.Div([
    html.Div(
        html.H1(
            children='ATP Statistics',
            style={'text-align': 'center', 'color': 'mediumblue', 'font-family': 'Arial', 'font-weight': 'bold'},
            className='content-container'
            )
    )],
    className='header',
)


markdown = """
**Data Attribution:** The data used here is (part of) the amazing dataset created by [**Jeff Sackmann**](http://www.jeffsackmann.com/) 
(Check out his [github repository](https://github.com/JeffSackmann/tennis_atp))

**Data Usage:** In particular, I am using atp tour-level main draw single matches from 1995 to present day. I am currently working towards an independent data gathering solution.

**Bug Fix:** This is a MVP which I had fun developing, mostly on weekends, for personal use. Therefore I am sure it is possible to find bugs and non-working interactions. 
If you find any or just want to get in touch with me, please feel free to reach out by [Linkedin](https://www.linkedin.com/in/francesco-bruzzesi/)

**Support:** I would love to grow the project, if you feel like supporting, you can [buy me a coffee ☕](https://www.buymeacoffee.com/fbruzzesi)
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
    *get_filter_rows(matches_df, players_df),
    html.Div(
        className='row',
        style={'margin-left': '2%', 'margin-right': '2%', 'margin-top':'1%'},
        children=[
        dcc.Tabs(id='tabs', 
            value='summary', 
            children=[
                dcc.Tab(label='Player Summary', value='summary', style=tab_style, selected_style=tab_style),
                dcc.Tab(label='Serve & Return', value='serve_return', style=tab_style, selected_style=tab_style),
                dcc.Tab(label='Under Pressure', value='under_pressure', style=tab_style, selected_style=tab_style),
                dcc.Tab(label='H2H', value='h2h', style=tab_style, selected_style=tab_style),
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


    
if __name__ == '__main__':
    app.run_server(
        debug=True
    )
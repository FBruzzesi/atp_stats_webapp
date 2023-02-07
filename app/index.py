from dash import dcc, html

# Local Imports
from app import app, server

# import callbacks

# from utils.filter_rows import get_filter_rows


tab_style = {"fontWeight": "bold"}

# # Load data
# data_path = os.getcwd() + '/data'

# tdl = TennisDataLoader(data_path=data_path)
# matches_df, players_df = callbacks.matches_df, callbacks.players_df


header = html.Div(
    [
        html.Div(
            html.H1(
                children="ATP Statistics",
                style={
                    "text-align": "center",
                    "color": "mediumblue",
                    "font-family": "Arial",
                    "font-weight": "bold",
                },
                className="content-container",
            )
        )
    ],
    className="header",
)

intro = open("intro.md", "r").read()

app.layout = html.Div(
    [
        header,
        html.Hr(style={"width": "96%", "margin-top": "1%", "margin-bottom": "1%"}),
        html.Details(
            open=True,
            children=[
                html.Summary(
                    id="open_details",
                    children="Close Description",
                    style={"margin-left": "1.5%"},
                ),
                dcc.Markdown(intro, style={"margin-left": "3%", "margin-top": "10pt"}),
                html.Div(id="open_state", children=True, style={"display": "none"}),
            ],
        ),
        # Hidden Div Block
        html.Div(
            [
                # Store selected player matches data
                html.Div(id="selected_player_matches", style={"display": "none"}),
                # Store selected player details data
                html.Div(id="selected_player_details", style={"display": "none"}),
                # Store selected player ranking data
                html.Div(id="selected_player_rank", style={"display": "none"}),
            ],
            style={"display": "none"},
        ),
        # *get_filter_rows(matches_df, players_df),
        html.Div(
            className="row",
            style={"margin-left": "2%", "margin-right": "2%", "margin-top": "1%"},
            children=[
                dcc.Tabs(
                    id="tabs",
                    value="summary",
                    children=[
                        dcc.Tab(
                            label="Player Summary",
                            value="summary",
                            style=tab_style,
                            selected_style=tab_style,
                        ),
                        dcc.Tab(
                            label="Serve & Return",
                            value="serve_return",
                            style=tab_style,
                            selected_style=tab_style,
                        ),
                        dcc.Tab(
                            label="Under Pressure",
                            value="under_pressure",
                            style=tab_style,
                            selected_style=tab_style,
                        ),
                        dcc.Tab(
                            label="H2H",
                            value="h2h",
                            style=tab_style,
                            selected_style=tab_style,
                        ),
                    ],
                    colors={
                        "border": "white",
                        "primary": "gold",
                        "background": "cornsilk",
                    },
                ),
                html.Div(id="tab-content"),
            ],
        ),
    ]
)


if __name__ == "__main__":
    app.run_server(debug=True, port=8000, host="127.0.0.1")

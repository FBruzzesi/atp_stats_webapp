import dash
import dash_bootstrap_components as dbc
from flask import Flask

# Flask server
server = Flask(__name__)

# Dash app
app = dash.Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True,
    title="ATP Stats",
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
    ],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"
    ],
    meta_tags=[
        {"charset": "utf-8"},
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, shrink-to-fit=no",
        },
    ],
)

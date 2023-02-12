from functools import partial

import dash_bootstrap_components as dbc
from dash import html

make_button = partial(dbc.Button, outline=True, external_link=True)

# Github link button
github = make_button(
    [html.I(className="bi bi-github"), " Github"],
    href="https://github.com/FBruzzesi/atp_stats_webapp",
    id="gh-link",
)

# Report a Bug link button
report_bug = make_button(
    [html.I(className="bi bi-bug-fill"), " Report a Bug"],
    href="https://github.com/FBruzzesi/atp_stats_webapp/issues",
    id="bug-link",
)

# Linkedin link button
linkedin = make_button(
    [html.I(className="bi bi-linkedin"), " Linkedin"],
    href="https://linkedin.com/in/francesco-bruzzesi/",
    id="linkedin-link",
)

# Support link button
support = make_button(
    [html.I(className="bi bi-cup-fill"), " Buy me a coffee"],
    href="https://ko-fi.com/francescobruzzesi",
    id="support-link",
)

spacing = dbc.Col(
    id="spacing",
    children=html.Div(),
    width={"size": 5},
)

title = dbc.Col(
    id="app-title",
    children=html.Div(
        [html.H3(["ATP Statistics"])],
        style={"margin-top": 20},
    ),
    width={"size": 2},
)

social = dbc.Col(
    id="social",
    children=html.Div(
        [github, report_bug, linkedin, support],
        style={"margin-top": 20, "justify": "right"},
    ),
    width={"size": 5},
)

# App header!
header = dbc.Row([spacing, title, social], id="app-header")

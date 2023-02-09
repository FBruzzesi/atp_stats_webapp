# Run it locally using:
# $ gunicorn --bind 0.0.0.0:8080 --pythonpath webapp index:server

import argparse
from dash import html

# Local imports
from app import app, server
import callbacks # TODO: Refactor callbacks before importing

from layout import header, layout


# App layout
app.layout = html.Div(
    id="layout",
    children=[
        header,
        html.Div(id="page-content", children=[layout]),
    ],
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--port", type=int, default=8080, help="server port")
    parser.add_argument(
        "-hs", "--host", type=str, default="0.0.0.0", help="server host"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=bool,
        default=True,
        help="whether or not run the server runs on debug mode",
    )

    args = parser.parse_args()

    app.run_server(
        debug=args.debug,
        host=args.host,
        port=args.port,
    )
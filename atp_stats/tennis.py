from datetime import date
from functools import reduce
from operator import and_
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from .utils import proportion_confint

surface_colors = {
    "Clay": "firebrick",
    "Grass": "seagreen",
    "Hard": "midnightblue",
    "Carpet": "limegreen",
}


class Player:
    """Create player by parsing matches and players dataframe"""

    def __init__(self, name: str, matches: pl.DataFrame, players: pl.DataFrame):
        """
        Arguments:
            player_name: name of the player
            matches: dataframe of all matches
            players: dataframe of all players
        """

        self.name: str = name

        player_info: Dict = players.filter(pl.col("player_name") == name).to_dicts()[0]

        self.pid: int = player_info["id"]

        self.matches: pl.DataFrame = matches.filter(pl.col("id") == self.pid).sort(
            ["tourney_date", "match_num"]
        )

        self.n_matches: int = self.matches.shape[0]

        self.ranks: pl.DataFrame = (
            self.matches.groupby("year")
            .agg([pl.col("rank").min().cast(pl.UInt16)])
            .sort("year")
        )

        self.info = {
            **player_info,
            **{
                "age": round((date.today() - player_info["birthdate"]).days / 365.25, 2),
                "best_rank": self.ranks.select(pl.min("rank")).to_dicts()[0]["rank"],
                "birthdate": player_info["birthdate"].strftime("%d %b %Y"),
            },
        }

    def __repr__(self):

        return f"Player {self.player_name} with {self.n_matches} matches played"


class FilteredPlayer:
    """WIP"""

    def __init__(
        self,
        player: Player,
        time_start: Optional[date] = date(1970, 1, 1),
        time_end: Optional[date] = date(2999, 12, 31),
        surfaces: Optional[List[str]] = None,
        tourney_levels: Optional[List[str]] = None,
        tournaments: Optional[List[str]] = None,
        opponents: Optional[List[str]] = None,
        rounds: Optional[List[str]] = None,
        opponent_ranks: Optional[int] = None,
    ):

        self.player = player

        filters = {
            "time_start": time_start,
            "time_end": time_end,
            "surface": surfaces,
            "tourney_level": tourney_levels,
            "tourney_name": tournaments,
            "opponent_name": opponents,
            "round": rounds,
            "opponent_rank": opponent_ranks,
        }

        self.selected_matches = self.filter_matches(self.player.matches, filters)
        self.n_matches = self.selected_matches.shape[0]
        self.selected_ranks = self.filter_ranks(self.selected_matches, self.player.ranks)

        self.stats_by_year = self.get_yearly_stats(self.selected_matches)
        self.lower_df, self.upper_df = self.yearly_confint(self.stats_by_year)
        self.surface_wl = self.get_surface_winloss(self.selected_matches)
        self.h2h = self.get_h2h(self.selected_matches)

        # TODO: Do we need this functionality? Probably deprecated!
        # self.get_overall_stats()

    @property
    def matches(self) -> pl.DataFrame:
        return self.player.matches

    @property
    def ranks(self) -> pl.DataFrame:
        return self.player.ranks

    @property
    def name(self) -> str:
        return self.player.name

    @property
    def info(self):
        return self.player.info

    @staticmethod
    def filter_matches(matches: pl.DataFrame, filters: Dict) -> pl.DataFrame:
        """
        Subsets matches based on dictionary of filters

        filter dict should at least contain time_start and time_end values

        """

        time_start = pl.col("tourney_date") > filters["time_start"]  # default 1970-01-01
        time_end = pl.col("tourney_date") < filters["time_end"]  # default 2999-12-31

        conditions = [time_start, time_end]

        for key, value in filters.items():
            if (key not in ["time_start", "time_end"]) and (value):

                m = (
                    (pl.col("opponent_rank") < value)
                    if key == "opponent_rank"
                    else (pl.col(key).is_in(value))
                )

                conditions.append(m)

        return matches.filter(reduce(and_, conditions))

    @staticmethod
    def filter_ranks(selected_matches: pl.DataFrame, ranks: pl.DataFrame) -> pl.DataFrame:
        """
        Generate time series of player rank and rank points
        """

        selected_ranks = (
            selected_matches.groupby("year")
            .agg(
                pl.col("winner")
                .filter(pl.col("round") == "F")
                .sum()
                .fill_null(0)
                .alias("tourney_won")
            )
            .join(ranks, on="year")
            .select(["year", "rank", "tourney_won"])
        )
        return selected_ranks

    # def get_overall_stats(self):

    #     m = self.selected_matches

    #     success_overall = pd.Series(m[self.success_cols].fillna(0).to_numpy().sum(axis=0),
    #                                 index=[f'success_{c}' for c in self.success_cols])
    #     total_overall = pd.Series(m[self.total_cols].to_numpy().sum(axis=0),
    #                               index=[f'total_{c}' for c in self.success_cols])

    #     self.success_overall = success_overall
    #     self.total_overall = total_overall

    #     self.success_overall['success_won'] = m['winner'].sum()
    #     self.total_overall['total_won'] = m.shape[0]

    #     self.perc_overall = pd.Series(100*self.success_overall.to_numpy()/self.total_overall.to_numpy(),
    #                     index=['perc_' + c.split('_')[-1] for c in self.success_overall.index]
    #                     )

    #     return self

    @staticmethod
    def _aggregate_stats(
        dframe: pl.DataFrame, level: Union[str, List[str]]
    ) -> pl.DataFrame:

        cols_to_sum = [
            "ace",
            "df",
            "svpt",
            "1stIn",
            "1stWon",
            "2ndIn",
            "2ndWon",
            "returnWon",
            "returnPlayed",
            "bpConverted",
            "bpTotal",
            "bpSaved",
            "bpFaced",
            "tbPlayed",
            "tbWon",
            "deciderPlayed",
            "deciderWon",
        ]

        agg_dframe = (
            dframe.groupby(level)
            .agg(
                [
                    pl.count("winner").alias("matches_played"),
                    pl.sum("winner").alias("matches_won"),
                ]
                + [pl.sum(c) for c in cols_to_sum]
            )
            .with_columns(
                [
                    (pl.col("matches_played") - pl.col("matches_won")).alias(
                        "matches_lost"
                    ),
                    (pl.col("matches_won") / pl.col("matches_played")).alias("win_rate"),
                    (pl.col("ace") / pl.col("svpt")).alias("perc_ace"),
                    (pl.col("df") / pl.col("svpt")).alias("perc_df"),
                    (pl.col("1stIn") / pl.col("svpt")).alias("perc_1stIn"),
                    (pl.col("1stWon") / pl.col("1stIn")).alias("perc_1stWon"),
                    (pl.col("2ndWon") / pl.col("2ndIn")).alias("perc_secondWon"),
                    (pl.col("returnWon") / pl.col("returnPlayed")).alias(
                        "perc_returnWon"
                    ),
                    (pl.col("bpConverted") / pl.col("bpTotal")).alias("perc_bpConverted"),
                    (pl.col("bpSaved") / pl.col("bpFaced")).alias("perc_bpSaved"),
                    (pl.col("tbPlayed") - pl.col("tbWon")).alias("tbLost"),
                    (pl.col("tbWon") / pl.col("tbPlayed")).alias("perc_tbWon"),
                    (pl.col("deciderPlayed") - pl.col("deciderWon")).alias("deciderLost"),
                    (pl.col("deciderWon") / pl.col("deciderPlayed")).alias(
                        "perc_decidingSetWon"
                    ),
                ]
            )
        )
        return agg_dframe

    @staticmethod
    def get_yearly_stats(selected_matches: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate statistics aggregated by year
        """

        stats_by_year = FilteredPlayer._aggregate_stats(
            selected_matches, level="year"
        ).sort("year")

        return stats_by_year

    def yearly_confint(
        self, stats_by_year: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:

        count = stats_by_year.select(
            pl.col(c).alias(str(i)) for i, c in enumerate(self.success_cols)
        ).to_numpy()
        nobs = stats_by_year.select(
            pl.col(c).alias(str(i)) for i, c in enumerate(self.total_cols)
        ).to_numpy()

        ci_low, ci_upp = proportion_confint(count, nobs)

        # TODO: Should we add the year or should we trust the ordering?
        lower_df = pl.DataFrame(ci_low, schema=[f"lower_{c}" for c in self.success_cols])
        upper_df = pl.DataFrame(ci_upp, schema=[f"upper_{c}" for c in self.success_cols])

        return lower_df, upper_df

    @staticmethod
    def get_surface_winloss(selected_matches: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate win/loss count by surface
        """
        return selected_matches.groupby(["surface", "result"]).agg(
            pl.count("id").alias("cnt")
        )

    @staticmethod
    def get_h2h(selected_matches: pl.DataFrame) -> pl.DataFrame:

        h2h = (
            FilteredPlayer._aggregate_stats(selected_matches, level="opponent_name")
            .sort("matches_played", reverse=True)
            .head(15)  # 15 most frequent opponents
        )

        return h2h

    @property
    def success_cols(self) -> List[str]:
        # TODO: move outside
        return [
            "ace",
            "df",
            "1stIn",
            "1stWon",
            "2ndWon",
            "returnWon",
            "bpConverted",
            "bpSaved",
            "tbWon",
            "deciderWon",
        ]

    @property
    def total_cols(self) -> List[str]:
        # TODO: move outside
        return [
            "svpt",
            "svpt",
            "svpt",
            "1stIn",
            "2ndIn",
            "returnPlayed",
            "bpTotal",
            "bpFaced",
            "tbPlayed",
            "deciderPlayed",
        ]

    def __repr__(self):

        return f"{self.name}, number of matches: {self.n_matches}"


class Renderer:
    def __init__(self, player: FilteredPlayer):

        self.player = player

        self._colors = [
            "rgb(33,113,181)",
            "rgb(217,71,1)",
            "rgb(81, 178, 124)",
            "rgb(235, 127, 134)",
        ] * 2

    @property
    def selected_matches(self):
        return self.player.selected_matches

    @property
    def n_matches(self):
        return self.player.n_matches

    @property
    def selected_ranks(self):
        return self.player.selected_ranks

    @property
    def stats_by_year(self):
        return self.player.stats_by_year

    @property
    def lower_df(self):
        return self.player.lower_df

    @property
    def upper_df(self):
        return self.player.upper_df

    @property
    def surface_wl(self):
        return self.player.surface_wl

    @property
    def h2h(self):
        return self.player.h2h

    def plot_summary(self):

        fig1 = make_subplots(
            specs=[[{"secondary_y": True}]],
        )

        # Add Rank over time
        x1, y1, y2 = (
            self.selected_ranks.select(["year", "rank", "tourney_won"]).to_numpy().T
        )

        fig1.add_trace(
            go.Scatter(
                x=x1,
                y=y1,
                name="Rank",
                marker={"color": "goldenrod"},
                mode="lines+text",
                text=y1,
                textposition="bottom center",
                textfont_size=8,
                opacity=0.8,
            ),
            secondary_y=False,
            row=1,
            col=1,
        )

        fig1.add_trace(
            go.Bar(
                x=x1,
                y=y2,
                name="Tournaments Won",
                marker={"color": "midnightblue"},
                text=y2,
                textposition="inside",
                textfont_size=8,
                opacity=0.8,
            ),
            secondary_y=True,
            row=1,
            col=1,
        )

        fig1.update_layout(
            height=500,
            legend={
                "font": {"size": 10},
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.05,
                "xanchor": "right",
                "x": 1,
            },
            title_font_size=18,
            title={
                "text": "Best Rank and Titles by Year",
                "x": 0.5,
                "xanchor": "center",
                "y": 0.9,
                "yanchor": "top",
            },
            xaxis={"title": "Year-Month"},
            yaxis={"title": "Best Rank", "range": [np.max(y1) + 10, -2]},
            yaxis2={"title": "Tournaments Won", "range": [0, np.max(y2) * 1.1]},
        )

        fig2 = make_subplots(
            specs=[[{"secondary_y": True}]],
        )

        # Add Winrate over time

        x2, b1, b2, wr = (
            self.stats_by_year.select(
                ["year", "matches_won", "matches_lost", 100 * pl.col("win_rate")]
            )
            .to_numpy()
            .T
        )

        fig2.add_trace(
            go.Bar(
                x=x2,
                y=b1,
                name="Matches Won",
                marker={"color": "seagreen"},
                text=b1,
                textposition="inside",
                textfont_size=8,
                opacity=0.8,
            ),
            secondary_y=False,
            row=1,
            col=1,
        )
        fig2.add_trace(
            go.Bar(
                x=x2,
                y=b2,
                name="Matches Lost",
                marker={"color": "indianred"},
                text=b2,
                textposition="inside",
                textfont_size=8,
                opacity=0.8,
            ),
            secondary_y=False,
            row=1,
            col=1,
        )

        fig2.add_trace(
            go.Scatter(
                x=x2,
                y=wr,
                name="Win Rate",
                line={"color": "midnightblue", "width": 2},
                mode="lines+text",
                text=[str(p) + "%" for p in np.round(wr, 2)],
                textposition="top center",
                textfont_size=8,
            ),
            secondary_y=True,
            row=1,
            col=1,
        )

        fig2.update_layout(
            height=500,
            legend={
                "font": {"size": 10},
                "orientation": "h",
                "traceorder": "normal",
                "yanchor": "bottom",
                "y": 1.025,
                "xanchor": "right",
                "x": 1,
            },
            title_font_size=18,
            title={
                "text": "Win Rate and Played Matches by Year",
                "x": 0.5,
                "xanchor": "center",
                "y": 0.9,
                "yanchor": "top",
            },
            barmode="stack",
            xaxis={"title": "Year-Month"},
            yaxis1={
                "range": [
                    0,
                    int(self.stats_by_year.select(pl.max("matches_played")).to_numpy())
                    + 15,
                ],
                "title": "Number of Matches",
            },
            yaxis2={"range": [0, 105], "title": "Win Rate (%)"},
        )

        return fig1, fig2

    def plot_surface_wl(self, surface_colors: Dict[str, str] = surface_colors):

        fig = px.sunburst(
            data_frame=self.surface_wl.to_pandas(),
            path=["surface", "result"],
            values="cnt",
            names="cnt",
            hover_data=["cnt"],
        )

        fig.data[0].marker.colors = [
            surface_colors[s.split("/")[0]] for s in fig.data[0].ids
        ]

        fig.update_layout(
            title={
                "text": "Win-Loss by Surface",
                "y": 1,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
        )

        return fig

    def plot_serve_return_stats(self, columns: List[str]):

        y_dist = (
            100
            * self.selected_matches.select([f"perc_{c}" for c in columns]).to_numpy().T
        )
        x, *y_stats = (
            self.stats_by_year.select(
                ["year"] + [(pl.col(f"perc_{c}") * 100) for c in columns]
            )
            .to_numpy()
            .T
        )
        y_lower = 100 * self.lower_df.select([f"lower_{c}" for c in columns]).to_numpy().T
        y_upper = 100 * self.upper_df.select([f"upper_{c}" for c in columns]).to_numpy().T

        colors = self._colors

        n_cols, n_rows = 2, len(columns)

        subplot_titles = [
            [
                f"Percentage {c[0].upper() + c[1:]} and 95% CI by year",
                f"Single Match Perc. {c[0].upper() + c[1:]} Distribution",
            ]
            for c in columns
        ]

        fig = make_subplots(
            cols=n_cols,
            rows=n_rows,
            specs=[[{}, {}]] * n_rows,
            shared_xaxes=False,
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            subplot_titles=sum(subplot_titles, []),
            column_widths=[0.7, 0.3],
        )

        for i, col in enumerate(columns):

            # Upper bound
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_upper[i],
                    name=f"{col} upper bound",
                    fill=None,
                    mode="lines",
                    line=dict(color="darksalmon", width=1),
                ),
                row=i + 1,
                col=1,
            )

            # Lower Bound
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_lower[i],
                    name=f"{col} lower bound",
                    fill="tonexty",
                    mode="lines",
                    line=dict(color="darksalmon", width=1),
                ),
                row=i + 1,
                col=1,
            )

            # Realized
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_stats[i],
                    textposition="top center",
                    name=col,
                    mode="lines+markers",
                    connectgaps=True,
                    marker={"color": colors[i]},
                ),
                row=i + 1,
                col=1,
            )

            # Distribution
            c_dist = y_dist[i]
            hist, kde = ff.create_distplot(
                [c_dist],
                bin_size=(c_dist.max() - c_dist.min()) / 50,
                group_labels=[col],
                show_rug=False,
                colors=[colors[i]],
                histnorm="probability",
            )["data"]

            fig.add_trace(hist, row=i + 1, col=2)
            fig.add_trace(kde, row=i + 1, col=2)

        # Layout
        fig.update_layout(
            {
                "showlegend": False,
                "xaxis": {"title": "Year"},
                "yaxis": {"title": "Percentage"},
                **{f"xaxis{2*i+1}": {"title": "Year"} for i in range(1, n_rows + 1)},
                **{f"xaxis{2*i}": {"title": "Percentage"} for i in range(1, n_rows + 1)},
                **{
                    f"yaxis{2*i+1}": {"title": "Percentage"} for i in range(1, n_rows + 1)
                },
                **{
                    f"yaxis{2*i}": {"title": "Frequency", "side": "right"}
                    for i in range(1, n_rows + 1)
                },
            }
        )

        return fig

    def plot_boxplot_distribution(self, columns: List[str]):

        yrs, *y_dist = (
            self.selected_matches.select(
                ["year"] + [(pl.col(f"perc_{c}") * 100) for c in columns]
            )
            .to_numpy()
            .T
        )

        colors = self._colors

        n_cols, n_rows = 2, len(columns)
        subplot_titles = [[f"{c} Boxplot", f"{c} Distplot"] for c in columns]

        fig = make_subplots(
            cols=n_cols,
            rows=n_rows,
            specs=[[{}, {}]] * n_rows,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.02,
            subplot_titles=sum(subplot_titles, []),
            column_widths=[0.8, 0.2],
        )

        for i, col in enumerate(columns):

            fig.add_trace(
                go.Box(x=yrs, y=y_dist[i], marker_color=colors[i]),
                row=i + 1,
                col=1,
            )

            hist, kde = ff.create_distplot(
                [y_dist[i]],
                bin_size=1.5,
                group_labels=[col],
                show_rug=False,
                colors=[colors[i]],
                histnorm="probability",
            )["data"]

            hist_ = go.Histogram(
                y=hist["x"],
                histnorm="probability",
                ybins={
                    "start": hist["xbins"]["start"],
                    "end": hist["xbins"]["end"],
                    "size": hist["xbins"]["size"],
                },
                opacity=0.7,
                marker_color=colors[i],
            )

            kde_ = go.Scatter(x=kde["y"], y=kde["x"], marker_color=colors[i])

            fig.add_trace(hist_, row=i + 1, col=2)
            fig.add_trace(kde_, row=i + 1, col=2)

        # Layout
        fig.update_layout(
            {
                "showlegend": False,
                f"xaxis{n_rows*n_cols-1}": {"title": "Year"},
                f"xaxis{n_rows*n_cols}": {"title": "Frequency"},
                **{
                    f"yaxis{2*r-1}": {"title": "Percentage"} for r in range(1, n_rows + 1)
                },
            }
        )

        return fig

    def plot_under_pressure(self, columns: List[str]):

        x, *y_stats = (
            self.stats_by_year.select(
                ["year"] + [(pl.col(f"perc_{c}") * 100) for c in columns]
            )
            .to_numpy()
            .T
        )
        y_lower = 100 * self.lower_df.select([f"lower_{c}" for c in columns]).to_numpy().T
        y_upper = 100 * self.upper_df.select([f"upper_{c}" for c in columns]).to_numpy().T

        colors = self._colors

        n_cols, n_rows = 1, len(columns)
        subplot_titles = [
            f"Percentage {c[0].upper() + c[1:]} and 95% CI by year" for c in columns
        ]

        fig = make_subplots(
            cols=n_cols,
            rows=n_rows,
            specs=[[{}]] * n_rows,
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
            subplot_titles=subplot_titles,
        )

        for i, col in enumerate(columns):

            # Upper bound
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_upper[i],
                    name=f"{col} upper bound",
                    fill=None,
                    mode="lines",
                    line=dict(color="darksalmon", width=1),
                ),
                row=i + 1,
                col=1,
            )

            # Lower bound
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_lower[i],
                    name=f"{col} lower bound",
                    fill="tonexty",
                    mode="lines",
                    line=dict(color="darksalmon", width=1),
                ),
                row=i + 1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_stats[i],
                    textposition="top center",
                    name=col,
                    mode="lines+markers",
                    connectgaps=True,
                    marker={"color": colors[i]},
                ),
                row=i + 1,
                col=1,
            )

        # Layout
        fig.update_layout(
            {
                "showlegend": False,
                "xaxis": {"title": "Year"},
                "yaxis": {"title": "Percentage"},
                **{f"xaxis{i}": {"title": "Year"} for i in range(1, n_rows + 1)},
                **{f"yaxis{i}": {"title": "Percentage"} for i in range(1, n_rows + 1)},
            }
        )

        return fig

    def plot_h2h(self):

        x, b1, b2, wr = (
            self.h2h.select(
                ["opponent_name", "matches_won", "matches_lost", 100 * pl.col("win_rate")]
            )
            .to_numpy()
            .T
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=x,
                y=b1,
                name="Matches Won",
                marker={"color": "seagreen"},
                text=b1,
                textposition="inside",
                textfont_size=8,
                opacity=0.8,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=b2,
                name="Matches Lost",
                marker={"color": "indianred"},
                text=b2,
                textposition="inside",
                textfont_size=8,
                opacity=0.8,
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=wr,
                name="Win Rate",
                line={"color": "midnightblue", "width": 2},
                mode="markers+text",
                text=[str(p) + "%" for p in np.round(wr.astype(float), 2)],
                textposition="top center",
                textfont_size=8,
            ),
            secondary_y=True,
        )

        fig.update_layout(
            barmode="stack",
            legend={
                "font": {"size": 10},
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.05,
                "xanchor": "right",
                "x": 1,
            },
            title={
                "text": "Win Rate with most played opponents",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            xaxis={"type": "category", "title": "Opponent name"},
            yaxis={"range": [0, np.max(b1 + b2) + 15], "title": "Number of Matches"},
            yaxis2={"range": [0, 110], "title": "Win Rate (%)"},
        )

        return fig


# TODO: refactor plot_serve_return_stats and plot_under_pressure since they produce the same left fig

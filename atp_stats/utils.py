from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import stats


def get_player_name(full_name: str) -> str:
    """
    Abbreviates player name from "First Name Surname" to "F.N. Surname
    """
    name_split = full_name.split(" ")
    first_names = ".".join([e[0] for e in name_split[:-1]])
    last_name = name_split[-1]

    return ". ".join([first_names, last_name])


def proportion_confint(
    count: npt.ArrayLike, nobs: npt.ArrayLike, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fork of proportion_confint from statsmodels at
    https://www.statsmodels.org/dev/_modules/statsmodels/stats/proportion.html#proportion_confint

    Remark that various checks and transformation steps are skipped.
    Confidence interval for a binomial proportion using Wilson method

    Arguments
        count: number of successes (array must be integer values).
        nobs: total number of trials (array must contain integer values).
        alpha: Significance level, default 0.05. Must be in (0, 1)

    Returns
        ci_low: lower confidence level with coverage (approximately) 1-alpha.
        ci_upp: upper confidence level with coverage (approximately) 1-alpha.
    """
    q_ = count / nobs

    # method == "wilson"
    crit = stats.norm.isf(alpha / 2.0)
    crit2 = crit**2
    denom = 1 + crit2 / nobs
    center = (q_ + crit2 / (2 * nobs)) / denom
    dist = crit * np.sqrt(q_ * (1.0 - q_) / nobs + crit2 / (4.0 * nobs**2))
    dist /= denom
    ci_low = center - dist
    ci_upp = center + dist

    return ci_low, ci_upp

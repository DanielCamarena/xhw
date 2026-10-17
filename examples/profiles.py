# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Named example definitions, not exported constants or new library defaults."""

from _data import marine_definition

import xhw


def era5_profiles() -> dict:
    common = dict(
        phenomenon="ahw",
        day_definition="offset:-05:00",
        tail="upper",
        comparator=">",
        threshold_kind="percentile",
        absolute_threshold=None,
        min_duration_days=3,
        max_gap_days=0,
        join_rule="qualify_then_join",
        missing_policy="as_non_exceedance",
        season_months=None,
        category_scheme=None,
    )
    result = {}
    for variable, statistic in (("tx", "max"), ("tn", "min")):
        result[f"{variable.upper()}90"] = (
            xhw.DefinitionSpec(
                profile=f"perkins2013_{variable}90",
                variable=f"{variable}_daily",
                combination="single",
                daily_statistic=statistic,
                percentile=90.0,
                **common,
            ),
            variable,
        )
    for lag in (0, 1):
        result[f"TX+TN lag {lag}"] = (
            xhw.DefinitionSpec(
                profile="xhw_txtn90" if lag == 0 else "xhw_tx_then_tn90",
                variable="tx_daily&tn_daily",
                combination="and_daily",
                daily_statistic="native",
                percentile=None,
                components=(
                    xhw.DailyComponent(
                        variable="tx_daily",
                        daily_statistic="max",
                        percentile=90.0,
                        lag_days=0,
                    ),
                    xhw.DailyComponent(
                        variable="tn_daily",
                        daily_statistic="min",
                        percentile=90.0,
                        lag_days=lag,
                    ),
                ),
                **common,
            ),
            ("tx", "tn"),
        )
    return result


def example_profiles():
    """Return fresh, inspectable definitions using the existing publication API.

    Atmospheric examples use offset:-05:00 and daily TX/TN. Reference periods,
    units, grid and baseline remain separate explicit configurations.
    """
    air = era5_profiles()
    return {
        "mhw": marine_definition(),
        "mcs": marine_definition(cold=True),
        "ahw_tx90": air["TX90"][0],
        "ahw_tn90": air["TN90"][0],
        "ahw_txtn90": air["TX+TN lag 0"][0],
        "ahw_tx_then_tn90": air["TX+TN lag 1"][0],
    }

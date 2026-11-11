"""Reproduce hand calculations and Mumo outputs for two toy sensor scenarios.

The observations are illustrative examples, not measured production data or
evidence that any one metric is superior for a real deployment.
"""

from __future__ import annotations

import math
from typing import Dict, List

from mumo_framework import analyze_dataset


K = 1
ALPHA = 0.5
BETA = 0.5
GAMMA = 0.5
ALERT_TOLERANCE = 0.1


def manual_calculation(data: List[float]) -> Dict[str, float]:
    """Evaluate the documented formulas directly for a five-value example."""
    values = sorted(data)
    if len(values) != 5:
        raise ValueError("the hand-worked examples require exactly five values")

    top_mean = values[-1]
    top_remainder_mean = math.fsum(values[:-1]) / 4
    bottom_mean = values[0]
    bottom_remainder_mean = math.fsum(values[1:]) / 4

    p_mpu = ALPHA * top_mean + (1 - ALPHA) * top_remainder_mean
    p_mpd = ALPHA * bottom_mean + (1 - ALPHA) * bottom_remainder_mean
    p_gma = BETA * p_mpu + (1 - BETA) * p_mpd
    mbv = p_mpu - p_mpd

    core_mean = values[2]
    remainder_core_mean = math.fsum(values[:2] + values[3:]) / 4
    p_gmcsi = ALPHA * core_mean + (1 - ALPHA) * remainder_core_mean
    maei = GAMMA * p_gma + (1 - GAMMA) * p_gmcsi

    rest_core_mean = math.fsum(values[1:-1]) / 3
    p_mbts = ((top_mean - bottom_mean) / 2 + ALPHA * rest_core_mean) / 2

    return {
        "P-MPU": p_mpu,
        "P-MPD": p_mpd,
        "P-GMA": p_gma,
        "MBV": mbv,
        "P-GMCSI": p_gmcsi,
        "MAEI": maei,
        "P-MBTS": p_mbts,
    }


def show_scenario(title: str, observations: List[float]) -> None:
    hand = manual_calculation(observations)
    result = analyze_dataset(
        observations,
        k=K,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        compare_standard=True,
        alert_tolerance=ALERT_TOLERANCE,
    )

    print("\n" + title)
    print("Observations:", observations)
    print("Sorted:", sorted(observations))
    print("\nHand calculation:")
    for name, value in hand.items():
        print("  {:10s} {:.6g}".format(name + ":", value))

    print("\nLibrary calculation:")
    for name in hand:
        library_value = result.metrics[name]
        print("  {:10s} {:.6g}".format(name + ":", library_value))
        assert math.isclose(hand[name], library_value, rel_tol=1e-12, abs_tol=1e-12)

    if result.standard_metrics is None:
        raise AssertionError("standard metrics were requested but not returned")

    data_range = max(observations) - min(observations)
    threshold = ALERT_TOLERANCE * data_range
    comparisons = (
        (
            "Median / Core Truth (P-GMCSI)",
            result.standard_metrics["median"],
            result.metrics["P-GMCSI"],
        ),
        (
            "Mean / P-GMA",
            result.standard_metrics["mean"],
            result.metrics["P-GMA"],
        ),
        (
            "Population standard deviation / MBV (dispersion diagnostic)",
            result.standard_metrics["standard_deviation"],
            result.metrics["MBV"],
        ),
    )

    print("\nStandard metrics:")
    for name, value in result.standard_metrics.items():
        print("  {:24s} {:.6g}".format(name + ":", value))

    print(
        "\nComparison details (range={:.6g}, alert threshold={:.6g}, "
        "tolerance={:.0%}):".format(
            data_range, threshold, ALERT_TOLERANCE
        )
    )
    expected_alerts = []
    for name, standard_value, mumo_value in comparisons:
        difference = abs(standard_value - mumo_value)
        status = "ALERT" if difference > threshold else "within threshold"
        print(
            "  {}: standard={:.6g}, Mumo={:.6g}, difference={:.6g} "
            "({:.2%} of range) -> {}".format(
                name,
                standard_value,
                mumo_value,
                difference,
                difference / data_range if data_range else 0.0,
                status,
            )
        )
        if difference > threshold:
            expected_alerts.append(status)

    print("Blind-Spot Alerts:")
    if result.blind_spot_alerts:
        for alert in result.blind_spot_alerts:
            print("  " + alert)
    else:
        print("  None; all displayed differences are within the threshold.")

    assert len(result.blind_spot_alerts) == len(expected_alerts)
    print(
        "Explanation: a metric pair can differ and still show no alert when "
        "its absolute difference does not exceed {:.0%} of the data range."
        .format(ALERT_TOLERANCE)
    )


if __name__ == "__main__":
    show_scenario(
        "Scenario 1: hypothetical response-time sensor with a high spike",
        [10.0, 11.0, 12.0, 13.0, 100.0],
    )
    show_scenario(
        "Scenario 2: hypothetical output sensor with a low drop",
        [0.0, 10.0, 11.0, 12.0, 13.0],
    )

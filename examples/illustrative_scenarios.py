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

    print("\nStandard metrics:", result.standard_metrics)
    print("Blind-Spot Alerts:")
    if result.blind_spot_alerts:
        for alert in result.blind_spot_alerts:
            print("  " + alert)
    else:
        print("  None at the configured 10%-of-range threshold.")


if __name__ == "__main__":
    show_scenario(
        "Scenario 1: hypothetical response-time sensor with a high spike",
        [10.0, 11.0, 12.0, 13.0, 100.0],
    )
    show_scenario(
        "Scenario 2: hypothetical output sensor with a low drop",
        [0.0, 10.0, 11.0, 12.0, 13.0],
    )

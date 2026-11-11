"""Statistical boundary and risk-corridor calculations for the Mumo framework.

Based on the original Mumo Mathematical Framework created by Mumo Framework Authors (Root Project).
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import statistics
from collections.abc import Sized
from typing import Dict, Iterable, List, Optional, Sequence, SupportsFloat, Tuple, cast

__version__ = "1.1.0-alpha.1"

_HEAP_THRESHOLD = 2048


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError("{} must be a real number".format(name))
    try:
        result = float(cast(SupportsFloat, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("{} must be a real number".format(name)) from exc
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _validated_data(data: Iterable[float]) -> List[float]:
    if isinstance(data, (str, bytes)):
        raise TypeError("data must be an iterable of numbers")
    try:
        iterator = iter(data)
    except TypeError as exc:
        raise TypeError("data must be an iterable of numbers") from exc
    values = [_finite_float(value, "data value") for value in iterator]
    if len(values) < 2:
        raise ValueError("data must contain at least two values")
    return values


def _ordered_data(data: Iterable[float]) -> List[float]:
    return sorted(_validated_data(data))


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("{} must be an integer".format(name))
    if value <= 0:
        raise ValueError("{} must be greater than zero".format(name))
    return value


def _slice_size(k: int, n: int) -> int:
    value = _positive_integer(k, "k")
    if value >= n:
        raise ValueError("k must be smaller than the number of data values")
    return value


def _weight(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError("{} must be between 0 and 1".format(name))
    return result


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


class _AccurateSum:
    """Maintain a high-accuracy sum without retaining streamed values."""

    def __init__(self) -> None:
        self._partials: List[float] = []

    def add(self, value: float) -> None:
        partials = self._partials
        index = 0
        for partial in partials:
            if abs(value) < abs(partial):
                value, partial = partial, value
            high = value + partial
            low = partial - (high - value)
            if low:
                partials[index] = low
                index += 1
            value = high
        partials[index:] = [value]

    def total(self) -> float:
        return math.fsum(self._partials)


def _parametric_boundary_means(
    data: Iterable[float], k: int
) -> Tuple[float, float, float, float]:
    """Return top, top remainder, bottom, and bottom remainder means."""
    if isinstance(data, (str, bytes)):
        raise TypeError("data must be an iterable of numbers")
    if isinstance(data, Sized) and len(data) <= _HEAP_THRESHOLD:
        values = _ordered_data(data)
        size = _slice_size(k, len(values))
        return (
            _mean(values[-size:]),
            _mean(values[:-size]),
            _mean(values[:size]),
            _mean(values[size:]),
        )

    size = _positive_integer(k, "k")
    upper: List[float] = []
    lower: List[float] = []
    total = _AccurateSum()
    count = 0
    try:
        iterator = iter(data)
    except TypeError as exc:
        raise TypeError("data must be an iterable of numbers") from exc
    for raw_value in iterator:
        value = _finite_float(raw_value, "data value")
        count += 1
        total.add(value)
        if len(upper) < size:
            heapq.heappush(upper, value)
        elif value > upper[0]:
            heapq.heapreplace(upper, value)
        if len(lower) < size:
            heapq.heappush(lower, -value)
        elif value < -lower[0]:
            heapq.heapreplace(lower, -value)

    if count < 2:
        raise ValueError("data must contain at least two values")
    size = _slice_size(size, count)
    upper_mean = math.fsum(upper) / size
    lower_mean = -math.fsum(lower) / size
    total_value = total.total()
    upper_remainder = (total_value - math.fsum(upper)) / (count - size)
    lower_remainder = (total_value + math.fsum(lower)) / (count - size)
    return upper_mean, upper_remainder, lower_mean, lower_remainder


def calculate_mbts(data: Iterable[float]) -> float:
    """Return MBTS using the edge range and the mean of interior values."""
    values = _ordered_data(data)
    if len(values) < 3:
        raise ValueError("MBTS requires at least three data values")
    rest_mean = _mean(values[1:-1])
    return ((values[-1] - values[0]) / 2.0 + rest_mean) / 2.0


def calculate_gma_k(data: Iterable[float], k: int) -> float:
    """Return the non-parametric generalized average for boundary slices."""
    values = _ordered_data(data)
    size = _slice_size(k, len(values))
    return (
        _mean(values[-size:])
        + _mean(values[:-size])
        + _mean(values[:size])
        + _mean(values[size:])
    ) / 4.0


def calculate_p_mpu(data: Iterable[float], k: int, alpha: float = 0.5) -> float:
    """Return parametric upside, weighting the top-k slice by alpha."""
    weight = _weight(alpha, "alpha")
    top_mean, remainder_mean, _, _ = _parametric_boundary_means(data, k)
    return weight * top_mean + (1.0 - weight) * remainder_mean


def calculate_p_mpd(data: Iterable[float], k: int, alpha: float = 0.5) -> float:
    """Return parametric downside, weighting the bottom-k slice by alpha."""
    weight = _weight(alpha, "alpha")
    _, _, bottom_mean, remainder_mean = _parametric_boundary_means(data, k)
    return weight * bottom_mean + (1.0 - weight) * remainder_mean


def calculate_p_gma(
    data: Iterable[float],
    k: int,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """Blend parametric upside and downside using beta."""
    balance = _weight(beta, "beta")
    weight = _weight(alpha, "alpha")
    top_mean, top_remainder, bottom_mean, bottom_remainder = (
        _parametric_boundary_means(data, k)
    )
    upside = weight * top_mean + (1.0 - weight) * top_remainder
    downside = weight * bottom_mean + (1.0 - weight) * bottom_remainder
    return balance * upside + (1.0 - balance) * downside


def calculate_mbv(data: Iterable[float], k: int, alpha: float = 0.5) -> float:
    """Return MBV, the unit-preserving width between parametric boundaries."""
    weight = _weight(alpha, "alpha")
    top_mean, top_remainder, bottom_mean, bottom_remainder = (
        _parametric_boundary_means(data, k)
    )
    return weight * (top_mean - bottom_mean) + (1.0 - weight) * (
        top_remainder - bottom_remainder
    )


def calculate_p_mbts(
    data: Iterable[float], k: int, alpha: float = 0.5
) -> float:
    """Return parametric MBTS using edge slices and the remaining-data mean.

    The supplied formula's ``mu_rest_core`` is interpreted as the mean of
    values left after removing the bottom-k and top-k boundary slices.
    """
    values = _ordered_data(data)
    size = _slice_size(k, len(values))
    if 2 * size >= len(values):
        raise ValueError("P-MBTS requires at least one value outside both k-slices")
    weight = _weight(alpha, "alpha")
    upper_mean = _mean(values[-size:])
    lower_mean = _mean(values[:size])
    rest_core_mean = _mean(values[size:-size])
    return ((upper_mean - lower_mean) / 2.0 + weight * rest_core_mean) / 2.0


def calculate_p_gmcsi(
    data: Iterable[float], k: int, alpha: float = 0.5
) -> float:
    """Return P-GMCSI using the centered k-value core cluster.

    Values are sorted first. The core is the contiguous slice starting at
    ``floor((n - k) / 2)``; the remainder contains every value outside it.
    For fractional sizing, obtain ``k`` from
    ``autotune_parameters(data, fraction=p).k``.
    """
    values = _ordered_data(data)
    size = _slice_size(k, len(values))
    weight = _weight(alpha, "alpha")
    start = (len(values) - size) // 2
    core = values[start : start + size]
    remainder = values[:start] + values[start + size :]
    return weight * _mean(core) + (1.0 - weight) * _mean(remainder)


def calculate_maei(
    data: Iterable[float],
    k: int,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.5,
) -> float:
    """Blend the generalized average and core-squeeze index using gamma."""
    values = _ordered_data(data)
    balance = _weight(gamma, "gamma")
    generalized = calculate_p_gma(values, k, alpha, beta)
    core = calculate_p_gmcsi(values, k, alpha)
    return balance * generalized + (1.0 - balance) * core


def calculate_mchi(
    data: Iterable[float],
    k: int,
    alpha: float = 0.5,
    gamma: float = 0.5,
) -> float:
    """Blend parametric MBTS and the core-squeeze index using gamma."""
    values = _ordered_data(data)
    balance = _weight(gamma, "gamma")
    tension = calculate_p_mbts(values, k, alpha)
    core = calculate_p_gmcsi(values, k, alpha)
    return balance * tension + (1.0 - balance) * core


@dataclass(frozen=True)
class MumoParameters:
    """Autotuned parameters for a dataset."""

    k: int
    alpha: float
    beta: float
    gamma: float = 0.5


@dataclass(frozen=True)
class MumoAnalysis:
    """Formula results, optional standard metrics, and comparison alerts."""

    metrics: Dict[str, float]
    standard_metrics: Optional[Dict[str, float]]
    blind_spot_alerts: Tuple[str, ...]
    k: int
    alpha: float
    beta: float
    gamma: float
    routing: str


def autotune_parameters(
    data: Iterable[float],
    fraction: float = 0.05,
    mbv_baseline: Optional[float] = None,
    gamma: float = 0.5,
) -> MumoParameters:
    """Derive bounded k, alpha, and beta values from a dataset.

    The boundary/core size is ``floor(fraction * n)``, clamped to
    ``[1, n - 1]`` so both the selected slice and its remainder are nonempty.
    The specification's alpha and beta ratios are clamped to [0, 1]. If no
    MBV baseline is supplied, the observed data range is used. Constant data
    receives neutral alpha/beta settings where the corresponding ratio is
    undefined.

    When the returned ``k`` will also be used for P-MBTS, MAEI, or MCHI, it
    must additionally satisfy ``2 * k < n``; choose a fraction below 0.5
    for the usual fractional-sizing workflow.
    """
    values = _ordered_data(data)
    proportion = _finite_float(fraction, "fraction")
    if not 0.0 < proportion <= 1.0:
        raise ValueError("fraction must be greater than 0 and at most 1")
    balance = _weight(gamma, "gamma")

    size = min(len(values) - 1, max(1, math.floor(proportion * len(values))))
    span = values[-1] - values[0]
    average = _mean(values)
    if span == 0.0:
        alpha = 0.0
    elif average == 0.0:
        alpha = 1.0
    else:
        alpha = min(1.0, max(0.0, span / abs(average)))

    baseline = span if mbv_baseline is None else _finite_float(
        mbv_baseline, "mbv_baseline"
    )
    if baseline < 0.0:
        raise ValueError("mbv_baseline must not be negative")
    if baseline == 0.0:
        if span != 0.0:
            raise ValueError("mbv_baseline must be non-zero for non-constant data")
        beta = 0.5
    else:
        beta = min(1.0, max(0.0, (values[-1] - average) / baseline))
    return MumoParameters(k=size, alpha=alpha, beta=beta, gamma=balance)


def analyze_dataset(
    data: Iterable[float],
    k: Optional[int] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    fraction: float = 0.05,
    compare_standard: bool = False,
    alert_tolerance: float = 0.1,
) -> MumoAnalysis:
    """Calculate Mumo formulas and optionally compare standard metrics.

    Explicit ``k`` and weights override autotuning. Parametric boundary means
    use heaps for large datasets and unsized iterables; small sized datasets
    use a sorted-boundary calculation. The alert tolerance is a fraction of
    the data range.
    """
    values = _validated_data(data)
    if len(values) < 3:
        raise ValueError("analysis requires at least three data values")
    if not isinstance(compare_standard, bool):
        raise TypeError("compare_standard must be a boolean")
    tolerance = _weight(alert_tolerance, "alert_tolerance")

    tuned = autotune_parameters(values, fraction=fraction)
    size = (
        min(tuned.k, (len(values) - 1) // 2)
        if k is None
        else _slice_size(k, len(values))
    )
    if 2 * size >= len(values):
        raise ValueError("k must satisfy 2*k < the number of data values")
    weight = tuned.alpha if alpha is None else _weight(alpha, "alpha")
    balance = tuned.beta if beta is None else _weight(beta, "beta")
    equilibrium = tuned.gamma if gamma is None else _weight(gamma, "gamma")

    metrics = {
        "GMA-k": calculate_gma_k(values, size),
        "P-MPU": calculate_p_mpu(values, size, weight),
        "P-MPD": calculate_p_mpd(values, size, weight),
        "P-GMA": calculate_p_gma(values, size, weight, balance),
        "MBV": calculate_mbv(values, size, weight),
        "MBTS": calculate_mbts(values),
        "P-MBTS": calculate_p_mbts(values, size, weight),
        "P-GMCSI": calculate_p_gmcsi(values, size, weight),
        "MAEI": calculate_maei(values, size, weight, balance, equilibrium),
        "MCHI": calculate_mchi(values, size, weight, equilibrium),
    }

    standard_metrics: Optional[Dict[str, float]] = None
    alerts: Tuple[str, ...] = ()
    if compare_standard:
        standard_mean = _mean(values)
        standard_metrics = {
            "mean": standard_mean,
            "median": statistics.median(values),
            "standard_deviation": statistics.pstdev(values),
        }
        scale = values[-1] - values[0]
        references = (
            ("P-MPU", metrics["P-MPU"]),
            ("P-MPD", metrics["P-MPD"]),
            ("Core Truth (P-GMCSI)", metrics["P-GMCSI"]),
        )
        alerts = tuple(
            "Blind-Spot Alert: standard mean differs from {} by {:.6g} "
            "(>{:.0%} of data range).".format(
                name, abs(standard_mean - reference), tolerance
            )
            for name, reference in references
            if abs(standard_mean - reference) > tolerance * scale
        )

    return MumoAnalysis(
        metrics=metrics,
        standard_metrics=standard_metrics,
        blind_spot_alerts=alerts,
        k=size,
        alpha=weight,
        beta=balance,
        gamma=equilibrium,
        routing="heap" if len(values) > _HEAP_THRESHOLD else "sorted",
    )


def _time_values(times: Optional[Sequence[float]], count: int) -> List[float]:
    if times is None:
        return [float(index) for index in range(count)]
    if len(times) != count:
        raise ValueError("times and values must have the same length")
    result = [_finite_float(value, "time") for value in times]
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("times must be strictly increasing")
    return result


def risk_velocity(
    mbv_values: Iterable[float], times: Optional[Sequence[float]] = None
) -> List[float]:
    """Return interval risk velocities (first differences of MBV over time)."""
    values = [_finite_float(value, "MBV value") for value in mbv_values]
    if len(values) < 2:
        raise ValueError("at least two MBV values are required")
    time_values = _time_values(times, len(values))
    return [
        (right_value - left_value) / (right_time - left_time)
        for left_value, right_value, left_time, right_time in zip(
            values, values[1:], time_values, time_values[1:]
        )
    ]


def risk_acceleration(
    mbv_values: Iterable[float], times: Optional[Sequence[float]] = None
) -> List[float]:
    """Return acceleration estimates between adjacent risk-velocity intervals."""
    values = [_finite_float(value, "MBV value") for value in mbv_values]
    if len(values) < 3:
        raise ValueError("at least three MBV values are required")
    time_values = _time_values(times, len(values))
    velocity = risk_velocity(values, time_values)
    midpoints = [
        (left + right) / 2.0 for left, right in zip(time_values, time_values[1:])
    ]
    return [
        (right_velocity - left_velocity) / (right_time - left_time)
        for left_velocity, right_velocity, left_time, right_time in zip(
            velocity, velocity[1:], midpoints, midpoints[1:]
        )
    ]


def cumulative_exposure(
    floor_values: Iterable[float],
    observations: Iterable[float],
    times: Optional[Sequence[float]] = None,
) -> float:
    """Integrate floor minus observed values with the trapezoidal rule."""
    floors = [_finite_float(value, "floor value") for value in floor_values]
    spots = [_finite_float(value, "observation") for value in observations]
    if len(floors) != len(spots):
        raise ValueError("floor_values and observations must have the same length")
    if len(floors) < 2:
        raise ValueError("at least two points are required for integration")
    time_values = _time_values(times, len(floors))
    gaps = [floor - spot for floor, spot in zip(floors, spots)]
    return math.fsum(
        (left_gap + right_gap) * (right_time - left_time) / 2.0
        for left_gap, right_gap, left_time, right_time in zip(
            gaps, gaps[1:], time_values, time_values[1:]
        )
    )


def _matrix(values: Iterable[Iterable[float]], name: str) -> List[List[float]]:
    if isinstance(values, (str, bytes)):
        raise TypeError("{} must be a matrix".format(name))
    try:
        rows = [
            [_finite_float(value, "{} value".format(name)) for value in row]
            for row in values
        ]
    except TypeError as exc:
        raise TypeError("{} must be a matrix of numbers".format(name)) from exc
    if not rows or not rows[0]:
        raise ValueError("{} must not be empty".format(name))
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("{} must be rectangular".format(name))
    return rows


def boundary_matrix(
    data: Iterable[Iterable[float]], k: int, alpha: float = 0.5
) -> List[Tuple[float, float, float]]:
    """Calculate (P-MPU, P-MPD, MBV) for each row in a dataset matrix."""
    rows = _matrix(data, "data")
    return [
        (
            calculate_p_mpu(row, k, alpha),
            calculate_p_mpd(row, k, alpha),
            calculate_mbv(row, k, alpha),
        )
        for row in rows
    ]


def matrix_multiply(
    omega: Iterable[Iterable[float]], data: Iterable[Iterable[float]]
) -> List[List[float]]:
    """Multiply two compatible rectangular matrices (Y = Omega * X)."""
    left = _matrix(omega, "omega")
    right = _matrix(data, "data")
    if len(left[0]) != len(right):
        raise ValueError("omega columns must match data rows")
    columns = list(zip(*right))
    return [
        [math.fsum(a * b for a, b in zip(row, column)) for column in columns]
        for row in left
    ]


def _read_interactive_dataset() -> List[float]:
    print("Enter numbers separated by spaces or commas; use multiple lines.")
    print("Press Enter on a blank line when you have finished the dataset.")
    values: List[float] = []
    while True:
        line = input("data> ").strip()
        if not line:
            break
        try:
            line_values = [
                _finite_float(token, "data value")
                for token in line.replace(",", " ").split()
            ]
        except (TypeError, ValueError) as exc:
            print("Invalid data: {}. Please re-enter that line.".format(exc))
            continue
        values.extend(line_values)
    return values


def _read_interactive_float(prompt: str) -> float:
    while True:
        try:
            return _finite_float(input(prompt).strip(), prompt.strip())
        except (TypeError, ValueError) as exc:
            print("Invalid number: {}. Please try again.".format(exc))


def _read_interactive_int(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            return int(raw)
        except ValueError:
            print("Please enter a whole number.")


def _read_interactive_k(count: int) -> int:
    maximum = (count - 1) // 2
    while True:
        value = _read_interactive_int(
            "Boundary/core size k (1 to {}): ".format(maximum)
        )
        if 1 <= value <= maximum:
            return value
        print("For these formulas, k must be at least 1 and 2*k must be < n.")


def _read_interactive_weight(name: str) -> float:
    while True:
        value = _read_interactive_float("{} (0 to 1): ".format(name))
        try:
            return _weight(value, name)
        except ValueError as exc:
            print("{} Please try again.".format(exc))


def main() -> int:
    """Run an interactive prompt for calculating Mumo formulas."""
    print("Mumo Framework interactive calculator")
    print("Enter as many data values as needed; available memory is the limit.")
    try:
        while True:
            try:
                data = _read_interactive_dataset()
                values = _ordered_data(data)
                if len(values) < 3:
                    raise ValueError("at least three data values are required")
                k = _read_interactive_k(len(values))
                alpha = _read_interactive_weight("alpha")
                beta = _read_interactive_weight("beta")
                gamma = _read_interactive_weight("gamma")
                compare_standard = (
                    input("Compare standard metrics? [y/N]: ").strip().lower()
                    in ("y", "yes")
                )
                analysis = analyze_dataset(
                    values, k, alpha, beta, gamma, compare_standard=compare_standard
                )
                print("\nResults ({} values):".format(len(values)))
                for name, result in analysis.metrics.items():
                    print("{:10s} {:.10g}".format(name + ":", result))
                if analysis.standard_metrics is not None:
                    print("\nStandard baseline metrics:")
                    for name, result in analysis.standard_metrics.items():
                        print("{:22s} {:.10g}".format(name + ":", result))
                    for alert in analysis.blind_spot_alerts:
                        print(alert)
                print(
                    "Baseline routing: {} (k={}, alpha={:.4g})".format(
                        analysis.routing, analysis.k, analysis.alpha
                    )
                )
                print("Autotuned:", autotune_parameters(values))
            except (TypeError, ValueError) as exc:
                print("Could not calculate results: {}".format(exc))

            again = input("\nCalculate another dataset? [y/N]: ").strip().lower()
            if again not in ("y", "yes"):
                return 0
    except (EOFError, KeyboardInterrupt):
        print("\nExiting Mumo calculator.")
        return 0


class MumoEngine:
    """Convenient namespace exposing the framework's calculations."""

    analyze_dataset = staticmethod(analyze_dataset)
    calculate_mbts = staticmethod(calculate_mbts)
    calculate_gma_k = staticmethod(calculate_gma_k)
    calculate_p_mpu = staticmethod(calculate_p_mpu)
    calculate_p_mpd = staticmethod(calculate_p_mpd)
    calculate_p_gma = staticmethod(calculate_p_gma)
    calculate_mbv = staticmethod(calculate_mbv)
    calculate_p_mbts = staticmethod(calculate_p_mbts)
    calculate_p_gmcsi = staticmethod(calculate_p_gmcsi)
    calculate_maei = staticmethod(calculate_maei)
    calculate_mchi = staticmethod(calculate_mchi)
    autotune_parameters = staticmethod(autotune_parameters)
    risk_velocity = staticmethod(risk_velocity)
    risk_acceleration = staticmethod(risk_acceleration)
    cumulative_exposure = staticmethod(cumulative_exposure)
    boundary_matrix = staticmethod(boundary_matrix)
    matrix_multiply = staticmethod(matrix_multiply)


__all__ = [
    "MumoAnalysis",
    "MumoEngine",
    "MumoParameters",
    "__version__",
    "analyze_dataset",
    "autotune_parameters",
    "boundary_matrix",
    "calculate_gma_k",
    "calculate_maei",
    "calculate_mbv",
    "calculate_mbts",
    "calculate_mchi",
    "calculate_p_gma",
    "calculate_p_gmcsi",
    "calculate_p_mbts",
    "calculate_p_mpd",
    "calculate_p_mpu",
    "cumulative_exposure",
    "matrix_multiply",
    "risk_acceleration",
    "risk_velocity",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

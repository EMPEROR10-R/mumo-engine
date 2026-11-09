"""Statistical boundary and risk-corridor calculations for the Mumo framework.

Based on the original Mumo Mathematical Framework created by Mumo Framework Authors (Root Project).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Sequence, Tuple

__version__ = "1.0.0"


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError("{} must be a real number".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("{} must be a real number".format(name)) from exc
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _ordered_data(data: Iterable[float]) -> List[float]:
    if isinstance(data, (str, bytes)):
        raise TypeError("data must be an iterable of numbers")
    try:
        iterator = iter(data)
    except TypeError as exc:
        raise TypeError("data must be an iterable of numbers") from exc
    values = [_finite_float(value, "data value") for value in iterator]
    if len(values) < 2:
        raise ValueError("data must contain at least two values")
    return sorted(values)


def _positive_integer(value: int, name: str) -> int:
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


def calculate_mpu(data: Iterable[float]) -> float:
    """Return MPU: the average of the maximum and the remaining-data mean."""
    values = _ordered_data(data)
    return (values[-1] + _mean(values[:-1])) / 2.0


def calculate_mpd(data: Iterable[float]) -> float:
    """Return MPD: the average of the minimum and the remaining-data mean."""
    values = _ordered_data(data)
    return (values[0] + _mean(values[1:])) / 2.0


def calculate_ma(data: Iterable[float]) -> float:
    """Return the foundational Mumo average, (MPU + MPD) / 2."""
    return (calculate_mpu(data) + calculate_mpd(data)) / 2.0


def calculate_mbts(data: Iterable[float]) -> float:
    """Return MBTS using the edge range and the mean of interior values."""
    values = _ordered_data(data)
    if len(values) < 3:
        raise ValueError("MBTS requires at least three data values")
    rest_mean = _mean(values[1:-1])
    return ((values[-1] - values[0]) / 2.0 + rest_mean) / 2.0


def calculate_mpu_k(data: Iterable[float], k: int) -> float:
    """Return the unweighted generalized upside metric for the top k values."""
    values = _ordered_data(data)
    size = _slice_size(k, len(values))
    return (_mean(values[-size:]) + _mean(values[:-size])) / 2.0


def calculate_mpd_k(data: Iterable[float], k: int) -> float:
    """Return the unweighted generalized downside metric for the bottom k."""
    values = _ordered_data(data)
    size = _slice_size(k, len(values))
    return (_mean(values[:size]) + _mean(values[size:])) / 2.0


def calculate_gma_k(data: Iterable[float], k: int) -> float:
    """Return GMA for a boundary slice of size k."""
    return (calculate_mpu_k(data, k) + calculate_mpd_k(data, k)) / 2.0


def calculate_p_mpu(data: Iterable[float], k: int, alpha: float = 0.5) -> float:
    """Return parametric upside, weighting the top-k slice by alpha."""
    values = _ordered_data(data)
    size = _slice_size(k, len(values))
    weight = _weight(alpha, "alpha")
    return weight * _mean(values[-size:]) + (1.0 - weight) * _mean(values[:-size])


def calculate_p_mpd(data: Iterable[float], k: int, alpha: float = 0.5) -> float:
    """Return parametric downside, weighting the bottom-k slice by alpha."""
    values = _ordered_data(data)
    size = _slice_size(k, len(values))
    weight = _weight(alpha, "alpha")
    return weight * _mean(values[:size]) + (1.0 - weight) * _mean(values[size:])


def calculate_p_gma(
    data: Iterable[float],
    k: int,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """Blend parametric upside and downside using beta."""
    balance = _weight(beta, "beta")
    upside = calculate_p_mpu(data, k, alpha)
    downside = calculate_p_mpd(data, k, alpha)
    return balance * upside + (1.0 - balance) * downside


def calculate_mbv(data: Iterable[float], k: int, alpha: float = 0.5) -> float:
    """Return MBV, the unit-preserving width between parametric boundaries."""
    return calculate_p_mpu(data, k, alpha) - calculate_p_mpd(data, k, alpha)


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
    balance = _weight(gamma, "gamma")
    generalized = calculate_p_gma(data, k, alpha, beta)
    core = calculate_p_gmcsi(data, k, alpha)
    return balance * generalized + (1.0 - balance) * core


def calculate_mchi(
    data: Iterable[float],
    k: int,
    alpha: float = 0.5,
    gamma: float = 0.5,
) -> float:
    """Blend parametric MBTS and the core-squeeze index using gamma."""
    balance = _weight(gamma, "gamma")
    tension = calculate_p_mbts(data, k, alpha)
    core = calculate_p_gmcsi(data, k, alpha)
    return balance * tension + (1.0 - balance) * core


@dataclass(frozen=True)
class MumoParameters:
    """Autotuned parameters for a dataset."""

    k: int
    alpha: float
    beta: float
    gamma: float = 0.5


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
                k = _read_interactive_k(len(values))
                alpha = _read_interactive_weight("alpha")
                beta = _read_interactive_weight("beta")
                gamma = _read_interactive_weight("gamma")

                results = [
                    ("MPU", calculate_mpu(values)),
                    ("MPD", calculate_mpd(values)),
                    ("MA", calculate_ma(values)),
                    ("MBTS", calculate_mbts(values)),
                    ("MPU-k", calculate_mpu_k(values, k)),
                    ("MPD-k", calculate_mpd_k(values, k)),
                    ("GMA-k", calculate_gma_k(values, k)),
                    ("P-MPU", calculate_p_mpu(values, k, alpha)),
                    ("P-MPD", calculate_p_mpd(values, k, alpha)),
                    ("P-GMA", calculate_p_gma(values, k, alpha, beta)),
                    ("MBV", calculate_mbv(values, k, alpha)),
                    ("P-MBTS", calculate_p_mbts(values, k, alpha)),
                    ("P-GMCSI", calculate_p_gmcsi(values, k, alpha)),
                    ("MAEI", calculate_maei(values, k, alpha, beta, gamma)),
                    ("MCHI", calculate_mchi(values, k, alpha, gamma)),
                ]
                print("\nResults ({} values):".format(len(values)))
                for name, result in results:
                    print("{:10s} {:.10g}".format(name + ":", result))
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

    calculate_mpu = staticmethod(calculate_mpu)
    calculate_mpd = staticmethod(calculate_mpd)
    calculate_ma = staticmethod(calculate_ma)
    calculate_mbts = staticmethod(calculate_mbts)
    calculate_mpu_k = staticmethod(calculate_mpu_k)
    calculate_mpd_k = staticmethod(calculate_mpd_k)
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
    "MumoEngine",
    "MumoParameters",
    "__version__",
    "autotune_parameters",
    "boundary_matrix",
    "calculate_gma_k",
    "calculate_ma",
    "calculate_maei",
    "calculate_mbv",
    "calculate_mbts",
    "calculate_mchi",
    "calculate_mpd",
    "calculate_mpd_k",
    "calculate_mpu",
    "calculate_mpu_k",
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

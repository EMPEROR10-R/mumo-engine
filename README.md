# Mumo Framework Engine

**mumo-engine 1.1.0-alpha.2** is a Python library for parametric data
boundaries, consensus metrics, and risk-corridor calculations.

The active API centers on the parametric formulas and their generalized
counterparts. The original foundational formulas are retained below as
historical background only; they are not implemented or called by the package.

## Historical foundational tier

The first Mumo formulation described three unweighted formulas for a sorted
dataset `x_1 <= ... <= x_n`:

* **MPU (Mumo Potential Upside):**
  `(x_n + mean(x_1, ..., x_(n-1))) / 2`
* **MPD (Mumo Potential Downside):**
  `(x_1 + mean(x_2, ..., x_n)) / 2`
* **MA (Mumo Average):** `(MPU + MPD) / 2`

These formulas are documented as historical context, not part of the active
execution tier or public Python API. The current parametric baselines replace
the single-edge weighting with configurable boundary slices.

## Active formulas

For `n` observations, `k` boundary observations, and `alpha` in `[0, 1]`:

* **P-MPU:** `alpha * mean(top k) + (1 - alpha) * mean(all except top k)`
* **P-MPD:** `alpha * mean(bottom k) + (1 - alpha) * mean(all except bottom k)`
* **P-GMA:** `beta * P-MPU + (1 - beta) * P-MPD`
* **GMA-k:** the mean of the unweighted upper and lower generalized
  boundaries, each of which blends its `k`-slice mean with its remaining-data
  mean.
* **MBV:** `P-MPU - P-MPD`, the signed width of the parametric corridor.
* **P-GMCSI:** weighted mean of the centered contiguous `k`-value core and all
  values outside that core.
* **MAEI:** `gamma * P-GMA + (1 - gamma) * P-GMCSI`.
* **P-MBTS:** alpha-weighted tension value using the edge slices and the mean
  remaining after removing both boundary slices. It requires `2 * k < n`.

The parametric boundary functions accept an explicit `k` and `alpha`. For
sized inputs up to 2,048 values they use sorted boundary slices; larger inputs
and unsized iterables use bounded heaps for the primary parametric boundary
means. The heap-based `calculate_p_mpu`, `calculate_p_mpd`,
`calculate_p_gma`, and `calculate_mbv` functions consume an iterable once.
The full `analyze_dataset` result also includes formulas that require the full
dataset, such as core slicing.

## Self-calibration

`autotune_parameters(data, fraction=...)` returns `k`, `alpha`, `beta`, and
`gamma` in a `MumoParameters` value. The default fraction is 5%; `k` is
bounded so that both the slice and its remainder are nonempty. Alpha and beta
are derived from the observed range, mean, and optional MBV baseline, then
clamped to `[0, 1]`. When the returned `k` is used with P-MBTS or MAEI, it
must satisfy `2 * k < n`.

`analyze_dataset` uses those calibrated values when parameters are omitted.
Any explicit `k`, `alpha`, `beta`, or `gamma` takes precedence over
autotuning.

## Standard comparison and Blind-Spot Alerts

Set `compare_standard=True` in `analyze_dataset` to return the arithmetic
mean, median, and population standard deviation alongside the Mumo formulas.
Blind-Spot Alerts compare like aspects: standard median with **Core Truth**
(P-GMCSI), standard mean with P-GMA, and population standard deviation with
MBV as a **dispersion diagnostic**, not as a claim that the measures are
equivalent. MBV is a weighted boundary-corridor width; population standard
deviation describes distribution spread. Their difference is reported as a
fraction of the observed data range. An alert is emitted for each difference exceeding
`alert_tolerance * (max(data) - min(data))`; the default tolerance is 10% of
the observed range.

```python
from mumo_framework import analyze_dataset

data = [12.0, 15.2, 11.8, 45.0, 14.1, 13.9, 10.5]
analysis = analyze_dataset(
    data,
    k=2,                 # explicit overrides are preserved
    alpha=0.7,
    compare_standard=True,
)

print(analysis.metrics["P-MPU"])
print(analysis.standard_metrics)
for alert in analysis.blind_spot_alerts:
    print(alert)
```

`MumoAnalysis` contains `metrics`, optional `standard_metrics`,
`blind_spot_alerts`, the chosen parameters, and the primary boundary routing
(`"sorted"` or `"heap"`).

## Worked scenarios

[`examples/illustrative_scenarios.py`](./examples/illustrative_scenarios.py)
shows both hand calculations and matching Python results for two hypothetical
five-sample sensor scenarios: one with a high spike and one with a low drop.
It prints P-MPU, P-MPD, P-GMA, MBV, P-GMCSI, MAEI, P-MBTS, standard metrics,
and a comparison table with both values, their difference, the range-based
threshold, and whether an alert fired. Thus a “within threshold” result still
shows the calculated difference and why it did not alert. It also explains
that a small median/Core Truth or mean/P-GMA gap means those central estimates
are close relative to this sample's range, not that they are universally
equivalent. A small standard-deviation/MBV gap is only a numerical comparison:
standard deviation and MBV describe spread differently and are not
interchangeable. An alert means the gap exceeds the chosen screening
threshold; it does not establish which metric is more accurate. The threshold
is not a statistical significance test. The script asserts the hand
calculations against the library results and checks the alert count.

Run it from the project root after installing the project:

```shell
python -m pip install -e .
python examples/illustrative_scenarios.py
```

These are constructed examples for understanding formula behavior, not
validated production cases or proof that Mumo metrics outperform standard
statistics. A grant or research evaluation should add domain-specific,
appropriately sourced data, documented baselines, and reproducible evaluation.

## Advanced calculus and matrix extensions

* `risk_velocity(mbv_values, times=None)` calculates first differences per
  time interval.
* `risk_acceleration(mbv_values, times=None)` calculates differences between
  adjacent velocity intervals.
* `cumulative_exposure(floor_values, observations, times=None)` integrates
  `floor - observation` using the trapezoidal rule.
* `boundary_matrix(data, k, alpha=...)` calculates P-MPU, P-MPD, and MBV for
  each row.
* `matrix_multiply(omega, data)` calculates `omega * data` for compatible
  rectangular matrices.

## Installation and use

```shell
python -m pip install mumo-engine
```

Or install the checkout in editable mode:

```shell
python -m pip install -e .
```

```python
from mumo_framework import MumoEngine

data = [12.0, 15.2, 11.8, 45.0, 14.1, 13.9, 10.5]
p_mpu = MumoEngine.calculate_p_mpu(data, k=2, alpha=0.7)
p_mpd = MumoEngine.calculate_p_mpd(data, k=2, alpha=0.7)
mbv = MumoEngine.calculate_mbv(data, k=2, alpha=0.7)
print(p_mpu, p_mpd, mbv)
```

For an unsized stream, pass a known `k` to a primary boundary function:

```python
p_mpu = MumoEngine.calculate_p_mpu((read_value() for _ in range(100_000)), k=500)
```

Run the interactive calculator with `python -m mumo_framework`; it accepts
comma- or whitespace-separated values and can optionally compare standard
metrics. Run the desktop interface with `mumo-framework-gui` or
`python -m mumo_gui`.

## License

MIT. See [LICENSE](./LICENSE).

================================================================================
MUMO MATHEMATICAL FRAMEWORK ENGINE (mumo-engine)
================================================================================
Version: 1.0.0
Author: Mumo Framework Authors
License: MIT (LICENSE)

--------------------------------------------------------------------------------
1. EXECUTIVE SUMMARY & LIBRARY PURPOSE
--------------------------------------------------------------------------------
The Mumo Framework (`mumo-engine`) is an advanced mathematical computing 
library designed to replace legacy statistical metrics (such as standard 
deviation, raw arithmetic means, and isolated medians) with dynamic risk-corridor 
tracking, extreme boundary isolation, consensus core squeezing, autotuned parameter 
self-calibration, and multi-dimensional tensor risk modeling.

Traditional statistical tools suffer from critical real-world limitations:
- Standard deviation produces abstract squared variance units rather than 
  unit-accurate risk limits.
- Single outliers or sensor typos skew basic mean calculations.
- Single-point medians flip wildly in large streaming data.
- Manual tuning parameters introduce human operator bias.

The Mumo Engine solves these problems by establishing boundary envelopes, dynamic 
consensus squeezing, calculus-based derivative velocity tracking, and automated 
parameter self-calibration across massive datasets and GPU-accelerated matrices.


--------------------------------------------------------------------------------
2. PIPELINE ARCHITECTURE DIAGRAM
--------------------------------------------------------------------------------

                        [ THE TOTAL SC-PGMF PIPELINE ]
                                      |
         +----------------------------+----------------------------+
         |                            |                            |
         v                            v                            v
[ Boundary Modules ]         [ Core Stabilisers ]        [ Calculus Tensors ]
* P-MPU (Ceiling Limit)      * P-GMCSI (True Baseline)   * Derivatives (Velocity)
* P-MPD (Safety Floor)       * P-MBTS (Tension Weight)   * Integrals (Total Strain)
* MBV (Risk Corridor)        * MAEI (Master Engines)     * Matrices (Multi-Scale)


--------------------------------------------------------------------------------
3. COMPLETE FORMULA BREAKDOWN: FUNCTIONS, SYMBOLS & GAPS FILLED
--------------------------------------------------------------------------------

[SECTION A: FOUNDATIONAL MUMO MODEL (Small Sample Sizes: n, k = 1)]

1. Mumo Potential Upside (MPU)
   * Formula: MPU = ( x_n + (sum(x_i for i=1 to n-1) / (n-1)) ) / 2
   * Symbol Definitions:
       - x_n : The single maximum value in the sorted dataset X.
       - x_i : Individual baseline entries in the sorted dataset.
       - n   : Total number of data points.
   * Function: Evaluates maximum peak performance capability relative to baseline.
   * Gap Filled: Replaces raw maximum values (which are highly vulnerable to 
     single data entry errors or noise spikes) with a baseline-anchored peak limit.

2. Mumo Potential Downside (MPD)
   * Formula: MPD = ( x_1 + (sum(x_i for i=2 to n) / (n-1)) ) / 2
   * Symbol Definitions:
       - x_1 : The single minimum value in the sorted dataset X.
       - x_i : Individual baseline entries in the upper dataset.
       - n   : Total number of data points.
   * Function: Evaluates system risk floor safety relative to the remaining baseline.
   * Gap Filled: Eliminates raw minimum metrics by evaluating worst-case failure 
     risk in proportion to baseline health.

3. Mumo Average (MA)
   * Formula: MA = ( MPU + MPD ) / 2
   * Function: Establishes the central equilibrium point between MPU and MPD.
   * Gap Filled: Provides an unwarped balance point for small sample counts.


[SECTION B: GENERALIZED & PARAMETRIC BOUNDARY MODULES (Big Data & Slice k)]

4. Parametric Mumo Potential Upside (P-MPU)
   * Formula: P-MPU = alpha * (sum(top k) / k) + (1 - alpha) * (sum(remaining) / (n - k))
   * Symbol Definitions:
       - k     : Boundary slice size (number of top/bottom boundary elements).
       - alpha : Weighting slider controlling boundary sensitivity (0 <= alpha <= 1).
       - n     : Total size of the data array.
   * Function: Calculates continuous upper performance capacity across massive data streams.
   * Gap Filled: Prevents single data typos or corrupted streaming telemetry from 
     warping performance capacity estimates.

5. Parametric Mumo Potential Downside (P-MPD)
   * Formula: P-MPD = alpha * (sum(bottom k) / k) + (1 - alpha) * (sum(remaining) / (n - k))
   * Function: Establishes a flexible operational safety floor across large database streams.
   * Gap Filled: Stops transient system glitches from triggering false automated 
     emergency shutdown routines.

6. Parametric Generalized Mumo Average (P-GMA)
   * Formula: P-GMA = beta * (P-MPU) + (1 - beta) * (P-MPD)
   * Symbol Definitions:
       - beta : Risk-reward balance parameter slider (0 <= beta <= 1).
   * Function: Blends peak upside boundaries and risk floors according to operational goals.
   * Gap Filled: Enables real-time tuning of automated trading bots or server network 
     throttling based on target risk tolerance.

7. Mumo Bounds of Variance (MBV)
   * Formula: MBV = P-MPU - P-MPD
   * Function: Measures the physical width of the operational risk corridor.
   * Gap Filled: Replaces statistical standard deviation (which yields abstract 
     squared units) with a unit-accurate physical range measurement matching the input data.


[SECTION C: CORE STABILISERS]

8. Parametric Generalized Mumo Core-Squeeze Index (P-GMCSI)
   * Formula: P-GMCSI = alpha * (sum(Core Cluster C_k) / k) + (1 - alpha) * (sum(Remaining Mass) / (n - k))
   * Symbol Definitions:
       - C_k : The contiguous central k-value slice of the sorted dataset.
       - R_{n-k} : Every value outside C_k; this set contains n-k values.
       - Slice position: start at index floor((n-k)/2) in the sorted dataset.
   * Function: Squeezes middle elements to produce an uncorrupted central consensus value.
   * Gap Filled: Eliminates median flipping in large datasets while preserving full 
     consensus awareness of outer wing mass.

   For dynamic fractional sizing, set k = max(1, floor(p*n)) (capped at n-1)
   using autotune_parameters(data, fraction=p).k. This is the framework's
   fractional-size mode (equivalent to delta=1); omitting it leaves k under
   caller control (equivalent to delta=0). The calculation functions
   themselves remain deterministic and accept k as an ordinary parameter.
   When using k for P-MBTS, MAEI, or MCHI as well, choose p < 0.5 so
   2*k < n and P-MBTS has at least one value remaining between its two
   boundary slices.

9. Mumo Adaptive Equilibrium Index (MAEI)
   * Formula: MAEI = gamma * (P-GMA) + (1 - gamma) * (P-GMCSI)
   * Symbol Definitions:
       - gamma : Balance switch between outer threat boundary tracking and inner core stability.
   * Function: Master Engine unifying boundary limits and core consensus into one metric.
   * Gap Filled: Provides hardware engines with dynamic mathematical survival instincts.


[SECTION D: AUTOTUNE ZERO-BIAS ENGINE (SC-PGMF)]

10. Autotune Self-Calibration (SC-MPU, SC-MPD, SC-GMA, SC-GMCSI, SC-MAEI)
    * Formulas:
        k_auto     = floor(p * n)
        alpha_auto = (x_n - x_1) / mean(x)
        beta_auto  = (x_n - mean(x)) / MBV_baseline
    * Symbol Definitions:
        - p : Target fractional boundary depth (e.g., 0.05 for 5%).
        - delta, omega : Mode switches (0 = manual, 1 = fully automated autotune).
    * Function: Auto-derives k, alpha, and beta parameters straight from data volume and variance.
    * Gap Filled: Erases human parameter bias and eliminates manual recalibration during sudden market spikes.


[SECTION E: CALCULUS TENSORS & MULTI-DIMENSIONAL EXTENSIONS]

11. Risk Derivatives (Velocity & Acceleration)
    * Formulas:
        Velocity     = d(MBV)/dt = d(P-MPU)/dt - d(P-MPD)/dt
        Acceleration = d^2(MBV)/dt^2
    * Function: Tracks the rate of expansion and acceleration of system risk corridors over time vectors.
    * Gap Filled: Predicts market crashes or hardware failures before safety floors break by detecting positive acceleration.

12. Cumulative Risk Mass Integration (Total Exposure Volume)
    * Formula: Total Exposure Volume = integral_a^b ( P-MPD(t) - f(t) ) dt
    * Symbol Definitions:
        - f(t) : Continuous metric spot curve over time t from a to b.
    * Function: Integrates continuous structural strain volume over time windows.
    * Gap Filled: Quantifies cumulative physical fatigue or financial drawdown duration rather than static point snapshots.

13. Multi-Dimensional Risk Matrix (Tensor Surface Transformation)
    * Formula: Y = Omega * X
    * Symbol Definitions:
        - X     : High-dimensional data matrix of multi-asset streams.
        - Omega : Automated parameter state matrix.
    * Function: Executes multi-asset risk surface calculations simultaneously.
    * Gap Filled: Enables GPU hardware to compute thousands of asset risk metrics in a single clock cycle.


--------------------------------------------------------------------------------
4. INSTALLATION & QUICKSTART
--------------------------------------------------------------------------------
Install via PyPI:
    pip install mumo-engine

Python Example:
    from mumo_framework import MumoEngine

    data = [12.0, 15.2, 11.8, 45.0, 14.1, 13.9, 10.5]

    # Calculate Ceiling, Safety Floor, and Risk Corridor Width
    p_mpu = MumoEngine.calculate_p_mpu(data, k=2, alpha=0.7)
    p_mpd = MumoEngine.calculate_p_mpd(data, k=2, alpha=0.7)
    mbv   = MumoEngine.calculate_mbv(data, k=2, alpha=0.7)

    print(f"Performance Ceiling (P-MPU): {p_mpu:.2f}")
    print(f"Safety Floor (P-MPD):        {p_mpd:.2f}")
    print(f"Risk Corridor Width (MBV):   {mbv:.2f}")

Fractional core/boundary sizing:
    params = MumoEngine.autotune_parameters(data, fraction=0.10)
    p_gmcsi = MumoEngine.calculate_p_gmcsi(data, k=params.k, alpha=0.7)

For combined P-MBTS, MAEI, or MCHI calculations, use a fraction below 0.5
and a dataset of at least three values so the returned k satisfies 2*k < n.

Interactive calculator:
    python -m mumo_framework

The calculator accepts numbers separated by commas or spaces over as many
input lines as needed. Submit a blank line to finish the dataset, then enter
k, alpha, beta, and gamma. Its dataset size is limited by available memory.
P-MBTS uses the mean of the values remaining after removing the lowest k and
highest k entries as mu_rest_core; therefore, it requires 2*k < n.

Interactive desktop calculator:
    mumo-framework-gui

Or, when running from the source checkout:
    python -m mumo_gui

The desktop calculator lets you edit or paste a dataset and interactively adjust
k, alpha, beta, and gamma. Results update as the data or parameters change.
It uses Python's built-in Tkinter interface and adds no third-party dependencies.


--------------------------------------------------------------------------------
5. LICENSE
--------------------------------------------------------------------------------
This project is licensed under the MIT License. See the accompanying LICENSE
file for full terms.

Based on the original Mumo Mathematical Framework created by Mumo Framework Authors (Root Project).
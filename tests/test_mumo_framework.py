import math
import unittest

from mumo_framework import (
    MumoEngine,
    analyze_dataset,
    boundary_matrix,
    calculate_gma_k,
    calculate_mbv,
    calculate_p_gma,
    calculate_p_mpd,
    calculate_p_mpu,
    cumulative_exposure,
    matrix_multiply,
    risk_acceleration,
    risk_velocity,
)


class ParametricFormulaTests(unittest.TestCase):
    def test_primary_formulas_and_nonparametric_gma(self):
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(calculate_p_mpu(data, k=1, alpha=0.5), 3.0)
        self.assertEqual(calculate_p_mpd(data, k=1, alpha=0.5), 2.0)
        self.assertEqual(calculate_p_gma(data, k=1, alpha=0.5, beta=0.25), 2.25)
        self.assertEqual(calculate_mbv(data, k=1, alpha=0.5), 1.0)
        self.assertEqual(calculate_gma_k(data, k=1), 2.5)

    def test_heap_route_matches_sorted_reference(self):
        data = [float((index * 37) % 103) for index in range(3000)]
        ordered = sorted(data)
        k = 17
        alpha = 0.63
        top = math.fsum(ordered[-k:]) / k
        top_rest = math.fsum(ordered[:-k]) / (len(data) - k)
        bottom = math.fsum(ordered[:k]) / k
        bottom_rest = math.fsum(ordered[k:]) / (len(data) - k)
        self.assertAlmostEqual(
            calculate_p_mpu(data, k, alpha), alpha * top + (1 - alpha) * top_rest
        )
        self.assertAlmostEqual(
            calculate_p_mpd(data, k, alpha),
            alpha * bottom + (1 - alpha) * bottom_rest,
        )

    def test_analysis_routes_large_streams_through_heap_baselines(self):
        data = (float((index * 19) % 71) for index in range(2100))
        result = analyze_dataset(data, k=13, alpha=0.4)
        self.assertEqual(result.routing, "heap")

    def test_unsized_generator_is_consumed_once_for_blended_baseline(self):
        consumed = []

        def values():
            for value in [1.0, 2.0, 3.0, 4.0]:
                consumed.append(value)
                yield value

        self.assertEqual(calculate_p_gma(values(), k=1, alpha=0.5), 2.5)
        self.assertEqual(consumed, [1.0, 2.0, 3.0, 4.0])

    def test_analysis_comparison_alerts_and_parameter_overrides(self):
        result = analyze_dataset(
            [0.0, 0.0, 0.0, 0.0, 100.0],
            k=1,
            alpha=0.5,
            beta=0.25,
            gamma=0.75,
            compare_standard=True,
        )
        self.assertEqual((result.k, result.alpha, result.beta, result.gamma), (1, 0.5, 0.25, 0.75))
        self.assertEqual(result.standard_metrics["mean"], 20.0)
        self.assertEqual(result.standard_metrics["median"], 0.0)
        self.assertIn("P-MPU", result.blind_spot_alerts[0])
        self.assertEqual(result.routing, "sorted")

    def test_legacy_formulas_are_not_executable_public_api(self):
        for name in ("calculate_mpu", "calculate_mpd", "calculate_ma"):
            self.assertFalse(hasattr(MumoEngine, name))


class ExtensionTests(unittest.TestCase):
    def test_calculus_and_matrix_extensions(self):
        self.assertEqual(risk_velocity([1.0, 3.0, 7.0]), [2.0, 4.0])
        self.assertEqual(risk_acceleration([1.0, 3.0, 7.0]), [2.0])
        self.assertEqual(cumulative_exposure([5.0, 5.0], [2.0, 4.0]), 2.0)
        self.assertEqual(boundary_matrix([[1.0, 2.0, 3.0]], k=1)[0][0:2], (2.25, 1.75))
        self.assertEqual(matrix_multiply([[1.0, 2.0]], [[3.0], [4.0]]), [[11.0]])


if __name__ == "__main__":
    unittest.main()

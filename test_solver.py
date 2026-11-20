import unittest
from solver import SimulationConfig, analytical_solution, run_rk4_simulation


class TestNumericalSolver(unittest.TestCase):
    def test_boundary_alignment_and_length(self):
        cfg = SimulationConfig(steps=500, dt=0.01)
        times, pos, vel = run_rk4_simulation(cfg)
        self.assertEqual(len(times), 500)
        self.assertEqual(len(pos), 500)
        self.assertEqual(len(vel), 500)
        self.assertAlmostEqual(times[0], 0.0)
        self.assertAlmostEqual(times[-1], 4.99)

    def test_rk4_matches_analytical_baseline(self):
        cfg = SimulationConfig(steps=10000, dt=0.002)
        _, x_rk4, _ = run_rk4_simulation(cfg)
        _, x_exact = analytical_solution(cfg)
        max_error = max(abs(num - ref) for num, ref in zip(x_rk4, x_exact))
        self.assertLess(max_error, 1e-8)

    def test_invalid_parameters_raise_value_error(self):
        with self.assertRaises(ValueError):
            SimulationConfig(mass=-1.0).validate()


if __name__ == "__main__":
    unittest.main()

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SimulationConfig:
    mass: float = 1.0          # kg
    damping: float = 0.25      # N*s/m
    stiffness: float = 4.0     # N/m
    x0: float = 1.0            # initial displacement (m)
    v0: float = 0.0            # initial velocity (m/s)
    dt: float = 0.005          # time step (s)
    steps: int = 2000          # number of integration steps

    def validate(self) -> None:
        if self.mass <= 0 or self.stiffness <= 0:
            raise ValueError("Mass and stiffness must be strictly positive.")
        if self.damping < 0:
            raise ValueError("Damping coefficient cannot be negative.")
        if self.dt <= 0 or self.steps < 1:
            raise ValueError("Time step must be positive and steps >= 1.")


def analytical_solution(cfg: SimulationConfig) -> Tuple[List[float], List[float]]:
    """Exact closed-form solution for an underdamped harmonic oscillator."""
    cfg.validate()
    omega_0 = math.sqrt(cfg.stiffness / cfg.mass)
    gamma = cfg.damping / (2.0 * cfg.mass)
    if gamma >= omega_0:
        raise ValueError("Analytical helper supports underdamped regime only (gamma < omega_0).")

    omega_d = math.sqrt(omega_0**2 - gamma**2)
    times = [i * cfg.dt for i in range(cfg.steps)]
    positions = []

    for t in times:
        envelope = math.exp(-gamma * t)
        pos = envelope * (
            cfg.x0 * math.cos(omega_d * t)
            + ((cfg.v0 + gamma * cfg.x0) / omega_d) * math.sin(omega_d * t)
        )
        positions.append(float(pos))

    return times, positions


def run_rk4_simulation(cfg: SimulationConfig) -> Tuple[List[float], List[float], List[float]]:
    """4th-order Runge-Kutta numerical solver with exact n-1 boundary alignment."""
    cfg.validate()
    times = [0.0] * cfg.steps
    positions = [0.0] * cfg.steps
    velocities = [0.0] * cfg.steps

    x = float(cfg.x0)
    v = float(cfg.v0)
    dt = float(cfg.dt)

    def accel(pos: float, vel: float) -> float:
        return -(cfg.damping * vel + cfg.stiffness * pos) / cfg.mass

    for i in range(cfg.steps):
        times[i] = i * dt
        positions[i] = x
        velocities[i] = v

        k1_x = v
        k1_v = accel(x, v)

        k2_x = v + 0.5 * dt * k1_v
        k2_v = accel(x + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v)

        k3_x = v + 0.5 * dt * k2_v
        k3_v = accel(x + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v)

        k4_x = v + dt * k3_v
        k4_v = accel(x + dt * k3_x, v + dt * k3_v)

        x += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return times, positions, velocities


if __name__ == "__main__":
    config = SimulationConfig()
    t_rk4, x_rk4, _ = run_rk4_simulation(config)
    _, x_exact = analytical_solution(config)
    max_err = max(abs(a - b) for a, b in zip(x_rk4, x_exact))
    print(f"Completed {config.steps} steps | Max absolute error vs analytical: {max_err:.3e}")


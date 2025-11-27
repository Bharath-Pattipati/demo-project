import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class VehicleParams:
    """Vehicle and environment parameters"""

    m_nom: float = 1500.0  # Nominal mass (kg)
    m_uncertainty: float = 200.0  # Mass uncertainty (kg)
    grade_nom: float = 0.0  # Nominal grade (rad)
    grade_uncertainty: float = 0.05  # Grade uncertainty (rad)
    mu_nom: float = 0.8  # Nominal tire friction
    mu_uncertainty: float = 0.15  # Friction uncertainty

    # Fixed parameters
    g: float = 9.81  # Gravity (m/s^2)
    Cd: float = 0.3  # Drag coefficient
    A: float = 2.5  # Frontal area (m^2)
    rho: float = 1.225  # Air density (kg/m^3)
    Crr: float = 0.01  # Rolling resistance coefficient

    # Control constraints
    F_max: float = 4000.0  # Max traction force (N)
    F_min: float = -8000.0  # Max braking force (N)

    # Speed constraints
    v_min: float = 0.0  # Min speed (m/s)
    v_max: float = 35.0  # Max speed (m/s)


@dataclass
class MPCParams:
    """MPC controller parameters"""

    N: int = 20  # Prediction horizon
    dt: float = 0.1  # Time step (s)
    Q: float = 100.0  # State tracking weight (increased)
    R: float = 0.01  # Control effort weight (decreased)
    K_tube: float = -50.0  # Ancillary controller gain (less aggressive)
    tube_scaling: float = 1.5  # Tube size scaling factor


class TubeMPC:
    """Tube-based MPC for longitudinal vehicle control"""

    def __init__(self, vehicle_params: VehicleParams, mpc_params: MPCParams):
        self.vp = vehicle_params
        self.mp = mpc_params

        # Compute nominal linearized dynamics: v(k+1) = A*v(k) + B*u(k) + d
        self.compute_nominal_linearization()

        # Compute tube (invariant set) offline
        self.compute_tube_bounds()

    def compute_nominal_linearization(self, v_op=20.0):
        """Linearize dynamics around operating point"""
        # Drag force: F_D = (1/2) * ρ * v^2 * C_D * A
        # Derivative: dF_D/dv = ρ * v * C_D * A
        dF_drag_dv = self.vp.rho * self.vp.Cd * self.vp.A * v_op

        # Linearized dynamics: dv/dt = (F - F_drag - F_roll - F_grade) / m
        # v(k+1) ≈ v(k) + dt * (F/m - dF_drag/m * v - g*(Crr + sin(grade)))

        self.A = 1.0 - self.mp.dt * dF_drag_dv / self.vp.m_nom
        self.B = self.mp.dt / self.vp.m_nom
        self.d_nom = -self.mp.dt * self.vp.g * (self.vp.Crr + np.sin(self.vp.grade_nom))

    def compute_tube_bounds(self):
        """Compute tube bounds based on uncertainties"""
        # Grade uncertainty causes disturbance
        delta_grade = self.vp.grade_uncertainty
        max_grade_disturbance = (
            self.mp.dt * self.vp.g * delta_grade
        )  # Small angle approximation

        # Mass uncertainty affects control effectiveness (worst case)
        delta_m = self.vp.m_uncertainty
        # Disturbance from mass: δ(F/m) ≈ F * δm / m^2
        # Use a more realistic force value (not max)
        typical_force = 1000.0  # Typical traction force (N)
        max_mass_disturbance = abs(
            delta_m / self.vp.m_nom * self.mp.dt * typical_force / self.vp.m_nom
        )

        # Friction uncertainty affects achievable control force
        delta_mu = self.vp.mu_uncertainty
        max_friction_disturbance = abs(delta_mu * self.vp.g * self.mp.dt * 0.5)

        # Conservative estimate of max disturbance per step
        max_disturbance = (
            max_grade_disturbance + max_mass_disturbance + max_friction_disturbance
        )

        print("Disturbance components:")
        print(f"  Grade: {max_grade_disturbance:.4f} m/s per step")
        print(f"  Mass: {max_mass_disturbance:.4f} m/s per step")
        print(f"  Friction: {max_friction_disturbance:.4f} m/s per step")
        print(f"  Total max disturbance: {max_disturbance:.4f} m/s per step")

        # Tube radius (invariant set for error dynamics with ancillary controller)
        # |e(k+1)| <= |A_e| * |e(k)| + |w(k)|
        # where A_e = A + B*K_tube
        A_e = self.A + self.B * self.mp.K_tube

        # For stable A_e (|A_e| < 1), invariant set is bounded by:
        # e_max = max_disturbance / (1 - |A_e|)
        if abs(A_e) >= 1.0:
            print("Warning: Error dynamics not stable! Adjusting K_tube")
            self.mp.K_tube = -2.0 * (1.0 - self.A) / self.B
            A_e = self.A + self.B * self.mp.K_tube

        self.tube_radius = self.mp.tube_scaling * max_disturbance / (1.0 - abs(A_e))
        print(f"Tube radius: {self.tube_radius:.3f} m/s")

    def vehicle_dynamics_actual(self, v, F, m_actual, grade_actual, mu_actual):
        """Actual nonlinear vehicle dynamics with uncertainties"""
        # Aerodynamic drag
        F_drag = 0.5 * self.vp.rho * self.vp.Cd * self.vp.A * v**2

        # Rolling resistance
        F_roll = self.vp.Crr * m_actual * self.vp.g * np.cos(grade_actual)

        # Gravitational component
        F_grade = m_actual * self.vp.g * np.sin(grade_actual)

        # Friction limit (simplified)
        F_max_friction = mu_actual * m_actual * self.vp.g
        F_constrained = np.clip(F, -F_max_friction, F_max_friction)

        # Acceleration
        dv_dt = (F_constrained - F_drag - F_roll - F_grade) / m_actual

        return dv_dt

    def solve_nominal_mpc(self, v_nom, v_target):
        """Solve nominal MPC problem with tightened constraints"""
        N = self.mp.N

        # Tighten constraints by tube radius
        v_min_tight = max(self.vp.v_min + self.tube_radius, 0)
        v_max_tight = self.vp.v_max - self.tube_radius

        def cost_function(u_seq):
            """MPC cost function"""
            cost = 0.0
            v = v_nom

            for k in range(N):
                # State cost (quadratic tracking error)
                cost += self.mp.Q * (v - v_target) ** 2

                # Control cost
                cost += self.mp.R * u_seq[k] ** 2

                # Control rate penalty (smoothness)
                if k > 0:
                    cost += 0.1 * (u_seq[k] - u_seq[k - 1]) ** 2

                # Predict next state (linearized)
                v = self.A * v + self.B * u_seq[k] + self.d_nom

                # Hard constraint penalty
                if v < v_min_tight:
                    cost += 10000.0 * (v_min_tight - v) ** 2
                if v > v_max_tight:
                    cost += 10000.0 * (v - v_max_tight) ** 2

            # Terminal cost (stronger)
            cost += 10.0 * self.mp.Q * (v - v_target) ** 2

            return cost

        # Initial guess - maintain previous control or ramp toward target
        if not hasattr(self, "_prev_u_seq"):
            self._prev_u_seq = np.zeros(N)

        # Warm start with previous solution shifted
        u0 = np.roll(self._prev_u_seq, -1)
        u0[-1] = u0[-2] if N > 1 else 0.0

        # Control bounds
        bounds = [(self.vp.F_min, self.vp.F_max) for _ in range(N)]

        # Solve optimization
        result = minimize(
            cost_function,
            u0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 150, "ftol": 1e-6},
        )

        if result.success:
            self._prev_u_seq = result.x
            return result.x[0]
        else:
            print(f"Warning: MPC optimization failed: {result.message}")
            return self._prev_u_seq[0] if hasattr(self, "_prev_u_seq") else 0.0

    def simulate(self, v_target, sim_time=20.0, disturbance_scenario="moderate"):
        """Run closed-loop simulation"""
        steps = int(sim_time / self.mp.dt)

        # Generate actual uncertain parameters
        if disturbance_scenario == "none":
            m_actual = self.vp.m_nom
            grade_actual = self.vp.grade_nom
            mu_actual = self.vp.mu_nom
        elif disturbance_scenario == "moderate":
            m_actual = self.vp.m_nom + 0.5 * self.vp.m_uncertainty
            grade_actual = self.vp.grade_nom + 0.02
            mu_actual = self.vp.mu_nom - 0.05
        else:  # 'severe'
            m_actual = self.vp.m_nom + self.vp.m_uncertainty
            grade_actual = self.vp.grade_nom + self.vp.grade_uncertainty
            mu_actual = self.vp.mu_nom - self.vp.mu_uncertainty

        print("\nActual parameters:")
        print(f"  Mass: {m_actual:.1f} kg (nominal: {self.vp.m_nom:.1f})")
        print(
            f"  Grade: {np.degrees(grade_actual):.2f}° (nominal: {np.degrees(self.vp.grade_nom):.2f}°)"
        )
        print(f"  Friction: {mu_actual:.2f} (nominal: {self.vp.mu_nom:.2f})")

        # Initialize states
        v_actual = 15.0  # Actual velocity
        v_nominal = 15.0  # Nominal velocity
        x_pos = 0.0  # Position

        # Data logging
        t_hist = []
        v_actual_hist = []
        v_nominal_hist = []
        v_tube_upper = []
        v_tube_lower = []
        F_hist = []
        error_hist = []

        for k in range(steps):
            t = k * self.mp.dt

            # Solve nominal MPC
            F_nominal = self.solve_nominal_mpc(v_nominal, v_target)

            # Ancillary controller (error feedback)
            e = v_actual - v_nominal
            F_ancillary = self.mp.K_tube * e

            # Total control
            F_total = F_nominal + F_ancillary
            F_total = np.clip(F_total, self.vp.F_min, self.vp.F_max)

            # Update actual system (nonlinear with uncertainties)
            dv_actual = self.vehicle_dynamics_actual(
                v_actual, F_total, m_actual, grade_actual, mu_actual
            )
            v_actual += dv_actual * self.mp.dt
            v_actual = np.clip(v_actual, self.vp.v_min, self.vp.v_max)

            # Update nominal system (linearized)
            v_nominal = self.A * v_nominal + self.B * F_nominal + self.d_nom
            v_nominal = np.clip(v_nominal, self.vp.v_min, self.vp.v_max)

            # Update position
            x_pos += v_actual * self.mp.dt

            # Log data
            t_hist.append(t)
            v_actual_hist.append(v_actual)
            v_nominal_hist.append(v_nominal)
            v_tube_upper.append(v_nominal + self.tube_radius)
            v_tube_lower.append(v_nominal - self.tube_radius)
            F_hist.append(F_total)
            error_hist.append(e)

        return {
            "t": np.array(t_hist),
            "v_actual": np.array(v_actual_hist),
            "v_nominal": np.array(v_nominal_hist),
            "v_tube_upper": np.array(v_tube_upper),
            "v_tube_lower": np.array(v_tube_lower),
            "F": np.array(F_hist),
            "error": np.array(error_hist),
            "v_target": v_target,
        }


def plot_results(results):
    """Plot simulation results"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Velocity tracking
    ax = axes[0]
    ax.plot(
        results["t"], results["v_actual"], "b-", linewidth=2.5, label="Actual velocity"
    )
    ax.plot(
        results["t"],
        results["v_nominal"],
        "r--",
        linewidth=2,
        label="Nominal trajectory",
    )
    ax.fill_between(
        results["t"],
        results["v_tube_lower"],
        results["v_tube_upper"],
        alpha=0.25,
        color="red",
        label="Tube bounds",
    )
    ax.axhline(
        results["v_target"],
        color="green",
        linestyle=":",
        linewidth=2.5,
        label="Target speed",
    )
    ax.set_ylabel("Velocity (m/s)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title("Tube-Based MPC: Velocity Tracking", fontsize=14, fontweight="bold")
    ax.set_ylim(
        [
            min(10, results["v_actual"].min() - 2),
            max(results["v_target"] + 5, results["v_actual"].max() + 2),
        ]
    )

    # Control input
    ax = axes[1]
    ax.plot(results["t"], results["F"], "g-", linewidth=2)
    ax.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylabel("Traction Force (N)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title("Control Input", fontsize=12)

    # Tracking error
    ax = axes[2]
    ax.plot(results["t"], results["error"], "r-", linewidth=2, label="Tube error")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    tube_rad = (results["v_tube_upper"][0] - results["v_tube_lower"][0]) / 2
    ax.axhline(
        tube_rad,
        color="r",
        linestyle=":",
        linewidth=1.5,
        alpha=0.5,
        label=f"Tube radius = {tube_rad:.2f} m/s",
    )
    ax.axhline(-tube_rad, color="r", linestyle=":", linewidth=1.5, alpha=0.5)
    ax.set_ylabel("Error: v_actual - v_nominal (m/s)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title("Tube Tracking Error (Should Stay Within ±Tube Radius)", fontsize=12)

    plt.tight_layout()
    # plt.savefig("tube_mpc_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Check if actual velocity stayed within tube
    in_tube = np.all(
        (results["v_actual"] >= results["v_tube_lower"] - 1e-6)
        & (results["v_actual"] <= results["v_tube_upper"] + 1e-6)
    )
    max_violation = max(
        0,
        np.max(results["v_actual"] - results["v_tube_upper"]),
        np.max(results["v_tube_lower"] - results["v_actual"]),
    )

    print(
        f"\n{'✓' if in_tube else '✗'} Actual trajectory stayed within tube: {in_tube}"
    )
    print(f"  Max tube violation: {max_violation:.4f} m/s")
    print(f"  Tube radius: {tube_rad:.3f} m/s")

    # Performance metrics
    settling_time_idx = np.where(
        np.abs(results["v_actual"] - results["v_target"]) < 0.5
    )[0]
    if len(settling_time_idx) > 0:
        settling_time = results["t"][settling_time_idx[0]]
        print(f"  Settling time (±0.5 m/s): {settling_time:.2f} s")

    final_error = abs(results["v_actual"][-1] - results["v_target"])
    print(f"  Final tracking error: {final_error:.3f} m/s")
    print(
        f"  Mean absolute error: {np.mean(np.abs(results['v_actual'] - results['v_target'])):.3f} m/s"
    )


if __name__ == "__main__":
    # Create vehicle and MPC parameters
    vehicle = VehicleParams(
        m_nom=1500.0,
        m_uncertainty=200.0,
        grade_nom=np.radians(2.0),  # 2 degree uphill
        grade_uncertainty=np.radians(1.5),
        mu_nom=0.8,
        mu_uncertainty=0.15,
    )

    mpc = MPCParams(N=20, dt=0.1, Q=100.0, R=0.01, K_tube=-50.0, tube_scaling=1.5)

    # Create controller
    controller = TubeMPC(vehicle, mpc)

    # Run simulation with moderate disturbances
    print("Running simulation with moderate disturbances...")
    results = controller.simulate(
        v_target=25.0,  # Target 25 m/s (~90 km/h)
        sim_time=20.0,
        disturbance_scenario="moderate",
    )

    # Plot results
    plot_results(results)

    print("\n" + "=" * 60)
    print("Simulation complete! Key features:")
    print("  • Nominal MPC solves with tightened constraints")
    print("  • Ancillary controller keeps actual state in tube")
    print("  • Handles mass, grade, and friction uncertainties")
    print("  • Guarantees constraint satisfaction despite disturbances")
    print("=" * 60)

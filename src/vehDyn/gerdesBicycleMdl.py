import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")


@dataclass
class VehicleParams:
    """Vehicle parameters from Gerdes paper Table II"""

    m: float = 1725.0  # mass [kg]
    Iz: float = 1300.0  # yaw inertia [kg·m²]
    a: float = 1.35  # distance CG to front axle [m]
    b: float = 1.15  # distance CG to rear axle [m]
    Caf: float = 57.8e3  # front cornering stiffness [N/rad]
    Car: float = 110e3  # rear cornering stiffness [N/rad]
    Ux: float = 16.0  # longitudinal velocity [m/s]
    mu: float = 0.55  # friction coefficient
    g: float = 9.81  # gravity [m/s²]

    @property
    def Fz(self) -> float:
        """Normal force per axle [N]"""
        return self.m * self.g / 2


class SineWithDwellSteeringProfile:
    """Generates sine with dwell steering profile as shown in the figure"""

    def __init__(self, amplitude: float = 0.4, dwell_time: float = 0.5):
        self.amplitude = amplitude  # Peak steering angle (rad)
        self.dwell_time = dwell_time  # Dwell time at bottom (s)

        # Phase timing based on typical sine with dwell test
        self.t1 = 1.0  # Time to reach peak
        self.t2 = 2.0  # Time to reach bottom
        self.t3 = self.t2 + self.dwell_time  # End of dwell
        self.t4 = 5.0  # Return to zero

    def get_steering_angle(self, t: float) -> float:
        """Get steering angle at time t"""
        if t <= 0:
            return 0.0
        elif t <= self.t1:
            # Sine wave from 0 to peak
            return self.amplitude * np.sin((np.pi / 2) * (t / self.t1))
        elif t <= self.t2:
            # Sine wave from peak to -peak
            phase = np.pi / 2 + np.pi * ((t - self.t1) / (self.t2 - self.t1))
            return self.amplitude * np.sin(phase)
        elif t <= self.t3:
            # Dwell at bottom
            return -self.amplitude
        elif t <= self.t4:
            # Return to zero
            phase = -np.pi / 2 + (np.pi / 2) * ((t - self.t3) / (self.t4 - self.t3))
            return -self.amplitude * np.sin(phase)
        else:
            return 0.0


class BrushTireModel:
    """Brush tire model from Fiala/Pacejka as used in Gerdes paper"""

    def __init__(self, Ca: float, mu: float, Fz: float):
        self.Ca = Ca  # cornering stiffness [N/rad]
        self.mu = mu  # friction coefficient
        self.Fz = Fz  # normal load [N]

        # Saturation slip angle
        self.alpha_sat = np.arctan(3 * mu * Fz / Ca)

    def lateral_force(self, alpha: float) -> float:
        """Calculate lateral tire force using brush model (Eq. 4)"""
        # More aggressive clipping to prevent extreme values
        alpha = np.clip(alpha, -0.5, 0.5)  # Limit to ±28.6 degrees

        if abs(alpha) < self.alpha_sat:
            # Linear + nonlinear region
            tan_alpha = np.tan(alpha)
            Fy = (
                self.Ca * tan_alpha
                + (self.Ca**2 / (3 * self.mu * self.Fz)) * abs(tan_alpha) * tan_alpha
                - (self.Ca**3 / (27 * self.mu**2 * self.Fz**2)) * tan_alpha**3
            )
        else:
            # Saturated region
            Fy = self.mu * self.Fz * np.sign(alpha)

        # Additional safety clipping
        max_force = 1.2 * self.mu * self.Fz  # Allow 20% over friction limit
        return np.clip(Fy, -max_force, max_force)

    def linearized_stiffness(self, alpha_bar: float) -> Tuple[float, float]:
        """Get linearized stiffness and force offset at operating point alpha_bar"""
        # Clip operating point to reasonable range
        alpha_bar = np.clip(alpha_bar, -0.3, 0.3)

        if abs(alpha_bar) < self.alpha_sat:
            tan_alpha = np.tan(alpha_bar)
            C_bar = (
                self.Ca
                + (self.Ca**2 / (3 * self.mu * self.Fz)) * abs(tan_alpha)
                - (self.Ca**3 / (9 * self.mu**2 * self.Fz**2)) * tan_alpha**2
            )
            # Ensure positive stiffness
            C_bar = max(C_bar, 0.1 * self.Ca)
        else:
            C_bar = 0.1 * self.Ca  # Small stiffness in saturated region

        Fy_bar = self.lateral_force(alpha_bar)
        return C_bar, Fy_bar


class GerdesBicycleModel:
    """Planar bicycle model with linearized rear tire as in Gerdes paper"""

    def __init__(self, params: VehicleParams):
        self.params = params

        # Tire models
        self.front_tire = BrushTireModel(params.Caf, params.mu, params.Fz)
        self.rear_tire = BrushTireModel(params.Car, params.mu, params.Fz)

        # State: [beta, r, psi, s, e] (sideslip, yaw rate, heading, distance, lateral deviation)
        self.state = np.zeros(5)

        # History for better prediction
        self.alpha_r_history = []
        self.max_history = 10

    def get_slip_angles(
        self, beta: float, r: float, delta: float
    ) -> Tuple[float, float]:
        """Calculate front and rear slip angles (Eqs. 2-3)"""
        alpha_f = beta + self.params.a * r / self.params.Ux - delta
        alpha_r = beta - self.params.b * r / self.params.Ux

        # Clip slip angles to reasonable ranges
        alpha_f = np.clip(alpha_f, -0.5, 0.5)
        alpha_r = np.clip(alpha_r, -0.5, 0.5)

        return alpha_f, alpha_r

    def predict_rear_slip_angle(
        self, current_alpha_r: float, dt: float, horizon: float
    ) -> float:
        """Improved prediction of future rear slip angle"""
        # Update history
        self.alpha_r_history.append(current_alpha_r)
        if len(self.alpha_r_history) > self.max_history:
            self.alpha_r_history.pop(0)

        if len(self.alpha_r_history) < 3:
            return current_alpha_r

        # Use simple trend extrapolation with damping
        recent_values = np.array(self.alpha_r_history[-3:])
        trend = (recent_values[-1] - recent_values[0]) / (2 * dt)

        # Predict with exponential damping
        damping_factor = np.exp(-horizon / 0.5)  # 0.5s time constant
        predicted = current_alpha_r + trend * horizon * damping_factor

        # Clip prediction to reasonable range
        return np.clip(predicted, -0.3, 0.3)

    def dynamics_nonlinear_front(
        self,
        state: np.ndarray,
        delta: float,
        alpha_r_bar: float = 0.0,
        use_successive_linearization: bool = False,
    ) -> np.ndarray:
        """Vehicle dynamics with nonlinear front tire and linearized rear tire"""
        beta, r, psi, s, e = state

        # Clip states to reasonable ranges for stability
        beta = np.clip(beta, -0.5, 0.5)  # ±28.6 degrees
        r = np.clip(r, -2.0, 2.0)  # ±114 deg/s

        # Slip angles
        alpha_f, alpha_r = self.get_slip_angles(beta, r, delta)

        # Front tire force (nonlinear)
        Fyf = self.front_tire.lateral_force(alpha_f)

        # Rear tire force (linearized around operating point)
        if use_successive_linearization:
            C_bar, Fy_bar = self.rear_tire.linearized_stiffness(alpha_r_bar)
            # Improved linearization formula
            delta_alpha = alpha_r - alpha_r_bar
            # Limit the linearization error
            delta_alpha = np.clip(delta_alpha, -0.1, 0.1)
            Fyr = Fy_bar + C_bar * delta_alpha
        else:
            # Simple linear model
            Fyr = self.params.Car * alpha_r

        # Vehicle dynamics (Eqs. 1, 8-9)
        beta_dot = (Fyf + Fyr) / (self.params.m * self.params.Ux) - r
        r_dot = (self.params.a * Fyf - self.params.b * Fyr) / self.params.Iz

        # Stability check - add artificial damping if needed
        if abs(beta) > 0.3 or abs(r) > 1.5:
            damping_beta = -10.0 * beta  # Proportional damping
            damping_r = -5.0 * r
            beta_dot += damping_beta
            r_dot += damping_r

        # Path following dynamics (Eqs. 10-14)
        psi_dot = r
        e_dot = self.params.Ux * beta  # Small angle approximation
        s_dot = self.params.Ux  # Constant speed

        return np.array([beta_dot, r_dot, psi_dot, s_dot, e_dot])

    def simulate_trajectory(
        self,
        steering_profile: SineWithDwellSteeringProfile,
        dt: float = 0.01,
        total_time: float = 6.0,
        use_successive_linearization: bool = False,
        prediction_horizon: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate vehicle trajectory with given steering profile"""
        time_steps = int(total_time / dt)
        t_array = np.linspace(0, total_time, time_steps)

        # Initialize arrays
        states = np.zeros((time_steps, 5))
        delta_array = np.zeros(time_steps)
        alpha_r_predictions = np.zeros(time_steps)

        # Reset history
        self.alpha_r_history = []

        # Initial state
        state = np.zeros(5)
        states[0] = state

        for i in range(1, time_steps):
            t = t_array[i]
            delta = steering_profile.get_steering_angle(t)
            delta_array[i] = delta

            # Get current slip angles
            alpha_f, alpha_r = self.get_slip_angles(state[0], state[1], delta)

            # Predict future rear slip angle for successive linearization
            if use_successive_linearization:
                alpha_r_bar = self.predict_rear_slip_angle(
                    alpha_r, dt, prediction_horizon
                )
            else:
                alpha_r_bar = 0.0  # Linear model

            alpha_r_predictions[i] = alpha_r_bar

            # Integrate dynamics using RK4 with stability checks
            try:
                k1 = self.dynamics_nonlinear_front(
                    state, delta, alpha_r_bar, use_successive_linearization
                )
                k2 = self.dynamics_nonlinear_front(
                    state + dt / 2 * k1,
                    delta,
                    alpha_r_bar,
                    use_successive_linearization,
                )
                k3 = self.dynamics_nonlinear_front(
                    state + dt / 2 * k2,
                    delta,
                    alpha_r_bar,
                    use_successive_linearization,
                )
                k4 = self.dynamics_nonlinear_front(
                    state + dt * k3, delta, alpha_r_bar, use_successive_linearization
                )

                state_new = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                # Stability check
                if np.any(np.isnan(state_new)) or np.any(np.abs(state_new[:2]) > 10):
                    print(f"Warning: Simulation unstable at t={t:.2f}s, breaking")
                    states[i:] = states[i - 1]  # Hold last valid state
                    break

                state = state_new
                states[i] = state

            except Exception as e:
                print(f"Integration error at t={t:.2f}s: {e}")
                states[i:] = states[i - 1]
                break

        return t_array, states, delta_array


def evaluate_prediction_horizons():
    """Evaluate model performance with different prediction horizons"""
    params = VehicleParams()
    model = GerdesBicycleModel(params)
    steering_profile = SineWithDwellSteeringProfile(
        amplitude=0.3, dwell_time=2
    )  # Reduced amplitude

    # More conservative prediction horizons
    horizons = [0.1, 0.3, 0.5, 1.0, 1.5]
    total_time = 6.0

    # Simulate with different approaches
    print("Simulating trajectories with different prediction horizons...")

    # Linear rear tire model (baseline)
    print("  Running linear rear tire model...")
    t_linear, states_linear, delta_linear = model.simulate_trajectory(
        steering_profile, total_time=total_time, use_successive_linearization=False
    )

    # Successive linearization with different horizons
    results = {}
    for horizon in horizons:
        print(f"  Running successive linearization with horizon {horizon}s...")
        # Create new model instance to reset history
        model_horizon = GerdesBicycleModel(params)
        t, states, delta = model_horizon.simulate_trajectory(
            steering_profile,
            total_time=total_time,
            use_successive_linearization=True,
            prediction_horizon=horizon,
        )
        results[horizon] = (t, states, delta)

    return t_linear, states_linear, delta_linear, results, steering_profile


def plot_results(t_linear, states_linear, delta_linear, results, steering_profile):
    """Plot comparison results"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(
        "Fixed Gerdes Bicycle Model: Linear vs Successive Linearization Comparison",
        fontsize=14,
    )

    # Time vector for steering input
    t_steering = np.linspace(0, 6, 600)
    steering_input = [steering_profile.get_steering_angle(t) for t in t_steering]

    # Plot steering input
    axes[0, 0].plot(
        t_steering,
        np.degrees(steering_input),
        "k-",
        linewidth=2,
        label="Steering Input",
    )
    axes[0, 0].set_ylabel("Steering Angle [deg]")
    axes[0, 0].set_title("Sine with Dwell Steering Input")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot sideslip angle
    axes[0, 1].plot(
        t_linear,
        np.degrees(states_linear[:, 0]),
        "r--",
        linewidth=2,
        label="Linear Rear Tire",
    )
    colors = ["blue", "green", "orange", "purple", "cyan"]
    for i, (horizon, (t, states, _)) in enumerate(results.items()):
        axes[0, 1].plot(
            t,
            np.degrees(states[:, 0]),
            colors[i],
            linewidth=1.5,
            label=f"Successive Lin. (H={horizon}s)",
        )
    axes[0, 1].set_ylabel("Sideslip Angle β [deg]")
    axes[0, 1].set_title("Sideslip Angle Comparison")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot yaw rate
    axes[1, 0].plot(
        t_linear,
        np.degrees(states_linear[:, 1]),
        "r--",
        linewidth=2,
        label="Linear Rear Tire",
    )
    for i, (horizon, (t, states, _)) in enumerate(results.items()):
        axes[1, 0].plot(
            t,
            np.degrees(states[:, 1]),
            colors[i],
            linewidth=1.5,
            label=f"Successive Lin. (H={horizon}s)",
        )
    axes[1, 0].set_ylabel("Yaw Rate r [deg/s]")
    axes[1, 0].set_title("Yaw Rate Comparison")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot lateral deviation
    axes[1, 1].plot(
        t_linear, states_linear[:, 4], "r--", linewidth=2, label="Linear Rear Tire"
    )
    for i, (horizon, (t, states, _)) in enumerate(results.items()):
        axes[1, 1].plot(
            t,
            states[:, 4],
            colors[i],
            linewidth=1.5,
            label=f"Successive Lin. (H={horizon}s)",
        )
    axes[1, 1].set_ylabel("Lateral Deviation e [m]")
    axes[1, 1].set_title("Lateral Deviation Comparison")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # Phase portrait: sideslip vs yaw rate (like Figure 3 in paper)
    axes[2, 0].plot(
        np.degrees(states_linear[:, 0]),
        np.degrees(states_linear[:, 1]),
        "r--",
        linewidth=2,
        label="Linear Rear Tire",
    )
    for i, (horizon, (t, states, _)) in enumerate(results.items()):
        axes[2, 0].plot(
            np.degrees(states[:, 0]),
            np.degrees(states[:, 1]),
            colors[i],
            linewidth=1.5,
            label=f"H={horizon}s",
        )
    axes[2, 0].set_xlabel("Sideslip Angle β [deg]")
    axes[2, 0].set_ylabel("Yaw Rate r [deg/s")
    axes[2, 0].set_title("Phase Portrait (β-r)")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    # Trajectory plot
    axes[2, 1].plot(
        states_linear[:, 3],
        states_linear[:, 4],
        "r--",
        linewidth=2,
        label="Linear Rear Tire",
    )
    for i, (horizon, (t, states, _)) in enumerate(results.items()):
        axes[2, 1].plot(
            states[:, 3], states[:, 4], colors[i], linewidth=1.5, label=f"H={horizon}s"
        )
    axes[2, 1].set_xlabel("Distance s [m]")
    axes[2, 1].set_ylabel("Lateral Deviation e [m]")
    axes[2, 1].set_title("Vehicle Trajectory")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()

    plt.tight_layout()
    plt.show()


def analyze_tire_forces():
    """Analyze tire force characteristics"""
    params = VehicleParams()
    front_tire = BrushTireModel(params.Caf, params.mu, params.Fz)
    rear_tire = BrushTireModel(params.Car, params.mu, params.Fz)

    # Slip angle range
    alpha_range = np.linspace(-0.3, 0.3, 1000)

    # Calculate forces
    Fyf_nl = [front_tire.lateral_force(alpha) for alpha in alpha_range]
    Fyr_nl = [rear_tire.lateral_force(alpha) for alpha in alpha_range]

    # Linear approximations
    Fyf_linear = params.Caf * alpha_range
    Fyr_linear = params.Car * alpha_range

    # Plot tire characteristics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(
        np.degrees(alpha_range), Fyf_nl, "b-", linewidth=2, label="Nonlinear (Brush)"
    )
    ax1.plot(np.degrees(alpha_range), Fyf_linear, "b--", linewidth=2, label="Linear")
    ax1.axhline(
        y=params.mu * params.Fz,
        color="r",
        linestyle=":",
        alpha=0.7,
        label="Friction Limit",
    )
    ax1.axhline(y=-params.mu * params.Fz, color="r", linestyle=":", alpha=0.7)
    ax1.set_xlabel("Slip Angle [deg]")
    ax1.set_ylabel("Lateral Force [N]")
    ax1.set_title("Front Tire Characteristics")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(
        np.degrees(alpha_range), Fyr_nl, "g-", linewidth=2, label="Nonlinear (Brush)"
    )
    ax2.plot(np.degrees(alpha_range), Fyr_linear, "g--", linewidth=2, label="Linear")
    ax2.axhline(
        y=params.mu * params.Fz,
        color="r",
        linestyle=":",
        alpha=0.7,
        label="Friction Limit",
    )
    ax2.axhline(y=-params.mu * params.Fz, color="r", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Slip Angle [deg]")
    ax2.set_ylabel("Lateral Force [N]")
    ax2.set_title("Rear Tire Characteristics")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Print key parameters
    print("\nTire Model Parameters:")
    print(f"Front tire saturation angle: {np.degrees(front_tire.alpha_sat):.1f} deg")
    print(f"Rear tire saturation angle: {np.degrees(rear_tire.alpha_sat):.1f} deg")
    print(f"Friction limit: {params.mu * params.Fz:.0f} N")


def calculate_stability_metrics(states):
    """Calculate stability metrics"""
    beta = states[:, 0]  # sideslip
    r = states[:, 1]  # yaw rate

    # Maximum values
    beta_max = np.max(np.abs(beta))
    r_max = np.max(np.abs(r))

    # RMS values
    beta_rms = np.sqrt(np.mean(beta**2))
    r_rms = np.sqrt(np.mean(r**2))

    return {
        "beta_max_deg": np.degrees(beta_max),
        "r_max_deg": np.degrees(r_max),
        "beta_rms_deg": np.degrees(beta_rms),
        "r_rms_deg": np.degrees(r_rms),
    }


def main():
    """Main evaluation function"""
    print("Fixed Gerdes Bicycle Model Evaluation")
    print("=" * 40)

    # Analyze tire characteristics
    print("\n1. Analyzing tire force characteristics...")
    analyze_tire_forces()

    # Evaluate different prediction horizons
    print("\n2. Evaluating prediction horizons...")
    t_linear, states_linear, delta_linear, results, steering_profile = (
        evaluate_prediction_horizons()
    )

    # Plot results
    print("\n3. Plotting comparison results...")
    plot_results(t_linear, states_linear, delta_linear, results, steering_profile)

    # Calculate and compare stability metrics
    print("\n4. Stability Metrics Comparison:")
    print("-" * 40)

    # Linear model metrics
    linear_metrics = calculate_stability_metrics(states_linear)
    print("Linear Rear Tire Model:")
    print(f"  Max sideslip: {linear_metrics['beta_max_deg']:.2f} deg")
    print(f"  Max yaw rate: {linear_metrics['r_max_deg']:.2f} deg/s")
    print(f"  RMS sideslip: {linear_metrics['beta_rms_deg']:.2f} deg")
    print(f"  RMS yaw rate: {linear_metrics['r_rms_deg']:.2f} deg/s")

    # Successive linearization metrics
    for horizon, (t, states, _) in results.items():
        metrics = calculate_stability_metrics(states)
        print(f"\nSuccessive Linearization (H={horizon}s):")
        print(f"  Max sideslip: {metrics['beta_max_deg']:.2f} deg")
        print(f"  Max yaw rate: {metrics['r_max_deg']:.2f} deg/s")
        print(f"  RMS sideslip: {metrics['beta_rms_deg']:.2f} deg")
        print(f"  RMS yaw rate: {metrics['r_rms_deg']:.2f} deg/s")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

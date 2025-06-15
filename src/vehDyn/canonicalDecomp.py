import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict
from scipy.integrate import solve_ivp


@dataclass
class VehicleParams:
    """Vehicle parameters for bicycle model"""

    M: float = 1500.0  # Vehicle mass (kg)
    Izz: float = 2500.0  # Yaw moment of inertia (kg·m²)
    a: float = 1.2  # Distance from CG to front axle (m)
    b: float = 1.8  # Distance from CG to rear axle (m)
    Cf: float = 60000.0  # Front cornering stiffness (N/rad)
    Cr: float = 55000.0  # Rear cornering stiffness (N/rad)

    @property
    def L(self) -> float:
        """Wheelbase"""
        return self.a + self.b


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

    def get_steering_angle(self, time: float) -> float:
        """Get steering angle at given time following the profile shape"""
        if time < 0:
            return 0.0
        elif time < self.t1:
            # Rise to peak (quarter sine)
            return self.amplitude * np.sin(np.pi / 2 * time / self.t1)
        elif time < self.t2:
            # Fall from peak to negative peak (half sine)
            phase = np.pi / 2 + np.pi * (time - self.t1) / (self.t2 - self.t1)
            return self.amplitude * np.sin(phase)
        elif time < self.t3:
            # Dwell at negative peak
            return -self.amplitude
        elif time < self.t4:
            # Rise from negative peak to zero (quarter sine)
            phase = 3 * np.pi / 2 + np.pi / 2 * (time - self.t3) / (self.t4 - self.t3)
            return self.amplitude * np.sin(phase)
        else:
            return 0.0

    def get_steering_rate(self, time: float, dt: float = 1e-4) -> float:
        """Get steering rate using numerical differentiation"""
        return (
            self.get_steering_angle(time + dt) - self.get_steering_angle(time - dt)
        ) / (2 * dt)


class TraditionalBicycleModel:
    """Traditional bicycle model for ground truth simulation"""

    def __init__(self, vehicle_params: VehicleParams, vx: float):
        self.params = vehicle_params
        self.vx = vx

    def dynamics(self, t: float, state: np.ndarray, steering_profile) -> np.ndarray:
        """Bicycle model dynamics: [vy, r, Y, psi]"""
        vy, r, Y, psi = state

        # Get steering input
        delta = steering_profile.get_steering_angle(t)

        # Tire slip angles
        alpha_f = delta - (vy + self.params.a * r) / self.vx
        alpha_r = -(vy - self.params.b * r) / self.vx

        # Tire forces
        Fyf = self.params.Cf * alpha_f
        Fyr = self.params.Cr * alpha_r

        # Equations of motion
        vy_dot = (Fyf + Fyr) / self.params.M - r * self.vx
        r_dot = (self.params.a * Fyf - self.params.b * Fyr) / self.params.Izz
        Y_dot = vy + self.vx * psi
        psi_dot = r

        return np.array([vy_dot, r_dot, Y_dot, psi_dot])

    def simulate(
        self, t_span: tuple, initial_state: np.ndarray, steering_profile
    ) -> Dict:
        """Simulate the bicycle model"""
        sol = solve_ivp(
            lambda t, y: self.dynamics(t, y, steering_profile),
            t_span,
            initial_state,
            dense_output=True,
            rtol=1e-8,
        )
        return sol


class CanonicalDecomposition:
    """Canonical decomposition of bicycle model dynamics"""

    def __init__(self, vehicle_params: VehicleParams):
        self.params = vehicle_params
        self.A_trans = np.zeros((2, 2))
        self.C_coup = np.zeros((2, 2))
        self.B_ss = np.zeros(2)
        self.vx = 0.0

    def update_matrices(self, vx: float):
        """Update canonical decomposition matrices"""
        self.vx = vx

        # Transient dynamics matrix (diagonal damping terms)
        self.A_trans[0, 0] = -(self.params.Cf + self.params.Cr) / (self.params.M * vx)
        self.A_trans[1, 1] = -(
            self.params.a**2 * self.params.Cf + self.params.b**2 * self.params.Cr
        ) / (self.params.Izz * vx)
        self.A_trans[0, 1] = 0.0
        self.A_trans[1, 0] = 0.0

        # Coupling matrix (off-diagonal terms)
        self.C_coup[0, 0] = 0.0
        self.C_coup[0, 1] = (
            -(self.params.a * self.params.Cf - self.params.b * self.params.Cr)
            / (self.params.M * vx)
            - vx
        )
        self.C_coup[1, 0] = -(
            self.params.a * self.params.Cf - self.params.b * self.params.Cr
        ) / (self.params.Izz * vx)
        self.C_coup[1, 1] = 0.0

        # Steady-state input matrix
        Kus = (
            self.params.M
            * (self.params.b / self.params.Cf - self.params.a / self.params.Cr)
            / self.params.L**2
        )
        denominator = self.params.L + Kus * vx**2

        self.B_ss[0] = vx * self.params.b / denominator  # Lateral velocity gain
        self.B_ss[1] = vx / denominator  # Yaw rate gain

    def get_full_dynamics_matrix(self) -> np.ndarray:
        """Get complete dynamics matrix A = A_trans + C_coup"""
        return self.A_trans + self.C_coup


class CanonicalTrajectoryGenerator:
    """Trajectory generator using canonical decomposition"""

    def __init__(
        self, vehicle_params: VehicleParams, dt: float = 0.02, horizon: int = 50
    ):
        self.vehicle_params = vehicle_params
        self.dt = dt
        self.horizon = horizon
        self.canonical = CanonicalDecomposition(vehicle_params)

    def generate_steering_profile(
        self, delta_current: float, delta_rate_current: float, steering_input
    ) -> np.ndarray:
        """Generate future steering profile using exponential decay model"""
        profile = np.zeros(self.horizon + 1)
        profile[0] = delta_current

        tau_decay = 0.3  # Time constant for steering rate decay
        max_steering_angle = 0.6
        max_steering_rate = 5.0

        for i in range(1, self.horizon + 1):
            time_step = i * self.dt
            # Exponential decay of steering rate
            effective_rate = delta_rate_current * np.exp(-time_step / tau_decay)
            limited_rate = np.clip(
                effective_rate, -max_steering_rate, max_steering_rate
            )
            next_angle = profile[i - 1] + limited_rate * self.dt
            profile[i] = np.clip(next_angle, -max_steering_angle, max_steering_angle)

        return profile

    def generate_trajectory(
        self,
        x_current: np.ndarray,
        delta_current: float,
        delta_rate_current: float,
        vx: float,
        steering_input,
    ) -> Dict:
        """Generate canonical trajectory: steady-state + transient"""

        # Update canonical matrices
        self.canonical.update_matrices(vx)

        # Generate steering profile
        delta_profile = self.generate_steering_profile(
            delta_current, delta_rate_current, steering_input
        )

        # Initialize arrays
        traj_vy = np.zeros(self.horizon + 1)
        traj_r = np.zeros(self.horizon + 1)
        traj_vy_ss = np.zeros(self.horizon + 1)
        traj_r_ss = np.zeros(self.horizon + 1)
        traj_vy_trans = np.zeros(self.horizon + 1)
        traj_r_trans = np.zeros(self.horizon + 1)

        # Current state
        traj_vy[0] = x_current[0]
        traj_r[0] = x_current[1]

        # Initial steady-state and transient decomposition
        x_ss_initial = self.canonical.B_ss * delta_current
        e_trans = x_current - x_ss_initial

        # Store initial components
        traj_vy_ss[0] = x_ss_initial[0]
        traj_r_ss[0] = x_ss_initial[1]
        traj_vy_trans[0] = e_trans[0]
        traj_r_trans[0] = e_trans[1]

        # Get full dynamics matrix
        A_full = self.canonical.get_full_dynamics_matrix()

        # Evolve trajectory over horizon
        for i in range(1, self.horizon + 1):
            # Steady-state component (instantaneous response to steering)
            x_ss = self.canonical.B_ss * delta_profile[i]
            traj_vy_ss[i] = x_ss[0]
            traj_r_ss[i] = x_ss[1]

            # Transient component evolution
            e_trans_dot = A_full @ e_trans
            e_trans = e_trans + e_trans_dot * self.dt
            traj_vy_trans[i] = e_trans[0]
            traj_r_trans[i] = e_trans[1]

            # Total response = steady-state + transient
            traj_vy[i] = traj_vy_ss[i] + traj_vy_trans[i]
            traj_r[i] = traj_r_ss[i] + traj_r_trans[i]

        return {
            "vy_total": traj_vy,
            "r_total": traj_r,
            "vy_ss": traj_vy_ss,
            "r_ss": traj_r_ss,
            "vy_trans": traj_vy_trans,
            "r_trans": traj_r_trans,
            "steering_profile": delta_profile,
            "time_horizon": np.arange(self.horizon + 1) * self.dt,
        }


def showcase_canonical_vs_traditional():
    """Showcase canonical trajectory projections overlaid on traditional bicycle model"""

    # Setup
    vehicle = VehicleParams()
    vx = 25.0  # Vehicle speed (m/s)
    steering_profile = SineWithDwellSteeringProfile(amplitude=0.3, dwell_time=2)

    # Traditional bicycle model simulation
    bicycle_model = TraditionalBicycleModel(vehicle, vx)
    t_sim = (0, 6.0)
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])  # [vy, r, Y, psi]

    # Simulate ground truth
    sol = bicycle_model.simulate(t_sim, initial_state, steering_profile)
    t_truth = np.linspace(0, 6.0, 300)
    states_truth = sol.sol(t_truth)

    # Canonical trajectory generator
    traj_gen = CanonicalTrajectoryGenerator(vehicle, dt=0.02, horizon=50)

    # Time points for horizon projections
    projection_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    colors = plt.cm.tab10(np.linspace(0, 1, len(projection_times)))

    # Generate projections
    projections = []
    for i, t_proj in enumerate(projection_times):
        # Get current state from ground truth
        current_idx = np.argmin(np.abs(t_truth - t_proj))
        x_current = states_truth[:2, current_idx]  # [vy, r]

        # Get current steering
        delta_current = steering_profile.get_steering_angle(t_proj)
        delta_rate_current = steering_profile.get_steering_rate(t_proj)

        # Generate canonical prediction
        result = traj_gen.generate_trajectory(
            x_current, delta_current, delta_rate_current, vx, steering_profile
        )

        # Store with absolute time
        absolute_time = t_proj + result["time_horizon"]
        projections.append(
            {
                "start_time": t_proj,
                "absolute_time": absolute_time,
                "vy_total": result["vy_total"],
                "r_total": result["r_total"],
                "vy_ss": result["vy_ss"],
                "r_ss": result["r_ss"],
                "vy_trans": result["vy_trans"],
                "r_trans": result["r_trans"],
                "steering": result["steering_profile"],
                "color": colors[i],
            }
        )

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        "Canonical Decomposition Projections vs Traditional Bicycle Model\nSine with Dwell Test",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Lateral Velocity
    ax1 = axes[0]
    # Ground truth
    ax1.plot(
        t_truth,
        states_truth[0, :],
        "k-",
        linewidth=3,
        alpha=0.8,
        label="Traditional Bicycle Model (Ground Truth)",
    )

    # Canonical projections
    for proj in projections:
        ax1.plot(
            proj["absolute_time"],
            proj["vy_total"],
            color=proj["color"],
            linewidth=2,
            alpha=0.7,
            label=f"Canonical t={proj['start_time']:.1f}s",
        )

    ax1.set_ylabel("Lateral Velocity vy (m/s)")
    ax1.set_title("Lateral Velocity: Canonical Projections vs Ground Truth")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # Plot 2: Yaw Rate
    ax2 = axes[1]
    # Ground truth
    ax2.plot(
        t_truth,
        states_truth[1, :],
        "k-",
        linewidth=3,
        alpha=0.8,
        label="Traditional Bicycle Model (Ground Truth)",
    )

    # Canonical projections
    for proj in projections:
        ax2.plot(
            proj["absolute_time"],
            proj["r_total"],
            color=proj["color"],
            linewidth=2,
            alpha=0.7,
            label=f"Canonical t={proj['start_time']:.1f}s",
        )

    ax2.set_ylabel("Yaw Rate r (rad/s)")
    ax2.set_title("Yaw Rate: Canonical Projections vs Ground Truth")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # Plot 3: Steering Input
    ax3 = axes[2]
    t_steer = np.linspace(0, 6, 300)
    delta_steer = [steering_profile.get_steering_angle(t) for t in t_steer]
    ax3.plot(t_steer, delta_steer, "k-", linewidth=3, label="Steering Input")

    # Show projected steering profiles
    for proj in projections:
        ax3.plot(
            proj["absolute_time"],
            proj["steering"],
            color=proj["color"],
            linewidth=1.5,
            alpha=0.6,
            linestyle="--",
            label=f"Predicted t={proj['start_time']:.1f}s",
        )

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Steering Angle δ (rad)")
    ax3.set_title("Steering Input and Predicted Profiles")
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.82)

    # Print analysis
    print("Canonical Decomposition Analysis:")
    print(f"Vehicle speed: {vx} m/s")
    print(f"Prediction horizon: {traj_gen.horizon * traj_gen.dt:.1f} seconds")
    print(
        f"Steering test: Sine with dwell (amplitude: {steering_profile.amplitude:.1f} rad)"
    )

    # Calculate time constants
    traj_gen.canonical.update_matrices(vx)
    tau_vy = (
        -1.0 / traj_gen.canonical.A_trans[0, 0]
        if traj_gen.canonical.A_trans[0, 0] != 0
        else np.inf
    )
    tau_r = (
        -1.0 / traj_gen.canonical.A_trans[1, 1]
        if traj_gen.canonical.A_trans[1, 1] != 0
        else np.inf
    )

    print(f"Time constants - vy: {tau_vy:.3f}s, r: {tau_r:.3f}s")
    print(
        f"Steady-state gains - vy: {traj_gen.canonical.B_ss[0]:.3f}, r: {traj_gen.canonical.B_ss[1]:.3f}"
    )

    plt.show()

    return projections, states_truth, t_truth


if __name__ == "__main__":
    projections, ground_truth, time_truth = showcase_canonical_vs_traditional()

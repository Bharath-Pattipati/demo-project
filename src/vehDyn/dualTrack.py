import numpy as np
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.integrate import odeint


@dataclass
class VehicleParameters:
    """Vehicle parameters for dual-track model"""

    # Chassis parameters
    m: float = 1500.0  # Mass (kg)
    Iz: float = 2500.0  # Yaw moment of inertia (kg*m^2)
    lf: float = 1.2  # Distance from CG to front axle (m)
    lr: float = 1.3  # Distance from CG to rear axle (m)
    tf: float = 1.5  # Front track width (m)
    tr: float = 1.5  # Rear track width (m)
    h: float = 0.5  # CG height (m)

    # Wheel parameters
    Rw: float = 0.32  # Wheel radius (m)
    Iw: float = 1.5  # Wheel inertia (kg*m^2)

    # Pacejka MF6.1 lateral force parameters
    B_lat: float = 10.0  # Stiffness factor
    C_lat: float = 1.3  # Shape factor
    D_lat: float = 1.0  # Peak factor multiplier
    E_lat: float = -2.0  # Curvature factor

    # Pacejka MF6.1 longitudinal force parameters
    B_lon: float = 10.0
    C_lon: float = 1.65
    D_lon: float = 1.0
    E_lon: float = -1.0

    # Actuator dynamics
    tau_motor: float = 0.05  # Motor time constant (s)
    tau_brake: float = 0.03  # Brake time constant (s)

    # Limits
    T_motor_max: float = 400.0  # Max motor torque per wheel (Nm)
    T_brake_max: float = 2000.0  # Max brake torque per wheel (Nm)


class DualTrackVehicleModel:
    """
    Full planar dual-track vehicle dynamics model

    States (15):
        - vx, vy: Body velocities (m/s)
        - r: Yaw rate (rad/s)
        - omega_fl, omega_fr, omega_rl, omega_rr: Wheel speeds (rad/s) [4]
        - T_motor_fl, T_motor_fr, T_motor_rl, T_motor_rr: Motor torque states (Nm) [4]
        - T_brake_fl, T_brake_fr, T_brake_rl, T_brake_rr: Brake torque states (Nm) [4]

    Computed quantities (not states):
        - kappa_fl, kappa_fr, kappa_rl, kappa_rr: Wheel slip ratios [4]

    Inputs (9):
        - T_motor_cmd: [fl, fr, rl, rr] Motor torque commands (Nm)
        - T_brake_cmd: [fl, fr, rl, rr] Brake torque commands (Nm)
        - delta: Steering angle (rad)

    Disturbances (3):
        - wind_x, wind_y: Wind forces (N)
        - road_grade: Road inclination (rad)
    """

    def __init__(self, params: VehicleParameters = None):
        self.p = params if params else VehicleParameters()

    def pacejka_longitudinal(self, kappa: float, Fz: float) -> float:
        """
        Pacejka Magic Formula for longitudinal force

        Args:
            kappa: Longitudinal slip ratio
            Fz: Normal force (N)

        Returns:
            Fx: Longitudinal tire force (N)
        """
        p = self.p
        mu_peak = p.D_lon * Fz
        x = p.B_lon * kappa
        Fx = mu_peak * np.sin(p.C_lon * np.arctan(x - p.E_lon * (x - np.arctan(x))))
        return Fx

    def pacejka_lateral(self, alpha: float, Fz: float) -> float:
        """
        Pacejka Magic Formula for lateral force

        Args:
            alpha: Slip angle (rad)
            Fz: Normal force (N)

        Returns:
            Fy: Lateral tire force (N)
        """
        p = self.p
        mu_peak = p.D_lat * Fz
        x = p.B_lat * np.tan(alpha)
        Fy = mu_peak * np.sin(p.C_lat * np.arctan(x - p.E_lat * (x - np.arctan(x))))
        return Fy

    def compute_normal_loads(
        self, ax: float, ay: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute normal loads on each wheel considering load transfer

        Returns:
            Fz_fl, Fz_fr, Fz_rl, Fz_rr: Normal forces (N)
        """
        p = self.p
        g = 9.81

        # Static weight distribution
        Fz_f = p.m * g * p.lr / (p.lf + p.lr)
        Fz_r = p.m * g * p.lf / (p.lf + p.lr)

        # Longitudinal load transfer
        dFz_lon = p.m * ax * p.h / (p.lf + p.lr)
        Fz_f -= dFz_lon
        Fz_r += dFz_lon

        # Lateral load transfer (simplified)
        dFz_lat_f = p.m * ay * p.h / p.tf
        dFz_lat_r = p.m * ay * p.h / p.tr

        Fz_fl = Fz_f / 2 - dFz_lat_f / 2
        Fz_fr = Fz_f / 2 + dFz_lat_f / 2
        Fz_rl = Fz_r / 2 - dFz_lat_r / 2
        Fz_rr = Fz_r / 2 + dFz_lat_r / 2

        # Ensure positive normal loads
        Fz_fl = max(Fz_fl, 100.0)
        Fz_fr = max(Fz_fr, 100.0)
        Fz_rl = max(Fz_rl, 100.0)
        Fz_rr = max(Fz_rr, 100.0)

        return Fz_fl, Fz_fr, Fz_rl, Fz_rr

    def compute_wheel_velocities(
        self, vx: float, vy: float, r: float, delta: float
    ) -> dict:
        """
        Compute velocity components at each wheel contact patch

        Returns:
            Dictionary with wheel velocity components and slip angles
        """
        p = self.p

        # Wheel center velocities in body frame
        vx_fl = vx - r * p.tf / 2
        vy_fl = vy + r * p.lf

        vx_fr = vx + r * p.tf / 2
        vy_fr = vy + r * p.lf

        vx_rl = vx - r * p.tr / 2
        vy_rl = vy - r * p.lr

        vx_rr = vx + r * p.tr / 2
        vy_rr = vy - r * p.lr

        # Transform to wheel coordinates (front wheels are steered)
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        vx_fl_wh = cos_d * vx_fl + sin_d * vy_fl
        vy_fl_wh = -sin_d * vx_fl + cos_d * vy_fl

        vx_fr_wh = cos_d * vx_fr + sin_d * vy_fr
        vy_fr_wh = -sin_d * vx_fr + cos_d * vy_fr

        vx_rl_wh = vx_rl
        vy_rl_wh = vy_rl

        vx_rr_wh = vx_rr
        vy_rr_wh = vy_rr

        # Slip angles (angle between wheel heading and velocity vector)
        alpha_fl = -np.arctan2(vy_fl_wh, abs(vx_fl_wh) + 1e-6)
        alpha_fr = -np.arctan2(vy_fr_wh, abs(vx_fr_wh) + 1e-6)
        alpha_rl = -np.arctan2(vy_rl_wh, abs(vx_rl_wh) + 1e-6)
        alpha_rr = -np.arctan2(vy_rr_wh, abs(vx_rr_wh) + 1e-6)

        return {
            "vx": [vx_fl_wh, vx_fr_wh, vx_rl_wh, vx_rr_wh],
            "vy": [vy_fl_wh, vy_fr_wh, vy_rl_wh, vy_rr_wh],
            "alpha": [alpha_fl, alpha_fr, alpha_rl, alpha_rr],
        }

    def compute_slip_ratios(self, vx_wheels: list, omega_wheels: list) -> list:
        """
        Compute longitudinal slip ratios

        kappa = (omega * R - vx) / max(|vx|, |omega * R|)
        """
        p = self.p
        kappas = []

        for vx_wh, omega in zip(vx_wheels, omega_wheels):
            v_wheel = omega * p.Rw
            denominator = max(abs(vx_wh), abs(v_wheel), 0.1)  # Avoid division by zero
            kappa = (v_wheel - vx_wh) / denominator

            # Limit slip ratio to physical range
            kappa = np.clip(kappa, -1.0, 1.0)
            kappas.append(kappa)

        return kappas

    def dynamics(
        self, state: np.ndarray, t: float, control: dict, disturbances: dict
    ) -> np.ndarray:
        """
        State derivatives for the dual-track vehicle model

        State vector (15):
            [vx, vy, r,
             omega_fl, omega_fr, omega_rl, omega_rr,  [4]
             T_motor_fl, T_motor_fr, T_motor_rl, T_motor_rr,  [4]
             T_brake_fl, T_brake_fr, T_brake_rl, T_brake_rr]  [4]
        Total: 3 + 4 + 4 + 4 = 15 states
        """
        p = self.p

        # Extract states
        vx, vy, r = state[0:3]
        omega_fl, omega_fr, omega_rl, omega_rr = state[3:7]
        T_motor_fl, T_motor_fr, T_motor_rl, T_motor_rr = state[7:11]
        T_brake_fl, T_brake_fr, T_brake_rl, T_brake_rr = state[11:15]

        # Extract controls
        T_motor_cmd = control["T_motor_cmd"]  # [fl, fr, rl, rr]
        T_brake_cmd = control["T_brake_cmd"]  # [fl, fr, rl, rr]
        delta = control["delta"]

        # Extract disturbances
        wind_x = disturbances.get("wind_x", 0)
        wind_y = disturbances.get("wind_y", 0)
        road_grade = disturbances.get("road_grade", 0)

        # Compute normal loads (with load transfer)
        ax_approx = 0  # Will be computed iteratively in real implementation
        ay_approx = vx * r
        Fz_fl, Fz_fr, Fz_rl, Fz_rr = self.compute_normal_loads(ax_approx, ay_approx)
        Fz = [Fz_fl, Fz_fr, Fz_rl, Fz_rr]

        # Compute wheel velocities and slip angles
        wheel_vels = self.compute_wheel_velocities(vx, vy, r, delta)
        vx_wheels = wheel_vels["vx"]
        alphas = wheel_vels["alpha"]

        # Compute slip ratios algebraically (not integrated states)
        omegas = [omega_fl, omega_fr, omega_rl, omega_rr]
        kappas = self.compute_slip_ratios(vx_wheels, omegas)

        # Compute tire forces using Pacejka model
        Fx_tires = [self.pacejka_longitudinal(k, fz) for k, fz in zip(kappas, Fz)]
        Fy_tires = [self.pacejka_lateral(alpha, fz) for alpha, fz in zip(alphas, Fz)]

        # Transform tire forces back to body frame
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        # Front wheels (steered)
        Fx_fl_body = cos_d * Fx_tires[0] - sin_d * Fy_tires[0]
        Fy_fl_body = sin_d * Fx_tires[0] + cos_d * Fy_tires[0]

        Fx_fr_body = cos_d * Fx_tires[1] - sin_d * Fy_tires[1]
        Fy_fr_body = sin_d * Fx_tires[1] + cos_d * Fy_tires[1]

        # Rear wheels (unsteered)
        Fx_rl_body = Fx_tires[2]
        Fy_rl_body = Fy_tires[2]

        Fx_rr_body = Fx_tires[3]
        Fy_rr_body = Fy_tires[3]

        # Total forces and moments
        Fx_total = Fx_fl_body + Fx_fr_body + Fx_rl_body + Fx_rr_body + wind_x
        Fy_total = Fy_fl_body + Fy_fr_body + Fy_rl_body + Fy_rr_body + wind_y

        # Yaw moment
        Mz = (
            p.lf * (Fy_fl_body + Fy_fr_body)
            - p.lr * (Fy_rl_body + Fy_rr_body)
            + p.tf / 2 * (Fx_fr_body - Fx_fl_body)
            + p.tr / 2 * (Fx_rr_body - Fx_rl_body)
        )

        # Body dynamics
        dvx = Fx_total / p.m + r * vy - 9.81 * np.sin(road_grade)
        dvy = Fy_total / p.m - r * vx
        dr = Mz / p.Iz

        # Wheel dynamics
        T_motors = [T_motor_fl, T_motor_fr, T_motor_rl, T_motor_rr]
        T_brakes = [T_brake_fl, T_brake_fr, T_brake_rl, T_brake_rr]

        domega = []
        for i, (T_m, T_b, Fx_tire) in enumerate(zip(T_motors, T_brakes, Fx_tires)):
            T_net = T_m - T_b - Fx_tire * p.Rw
            domega.append(T_net / p.Iw)

        # Actuator dynamics (first-order)
        dT_motor = [
            (cmd - T_m) / p.tau_motor for cmd, T_m in zip(T_motor_cmd, T_motors)
        ]
        dT_brake = [
            (cmd - T_b) / p.tau_brake for cmd, T_b in zip(T_brake_cmd, T_brakes)
        ]

        # Assemble state derivative (15 states)
        dstate = np.array([dvx, dvy, dr, *domega, *dT_motor, *dT_brake])

        return dstate

    def simulate(
        self, x0: np.ndarray, t_span: np.ndarray, control_func, disturbance_func=None
    ):
        """
        Simulate the vehicle dynamics

        Args:
            x0: Initial state (15,)
            t_span: Time vector
            control_func: Function that returns control dict given (state, t)
            disturbance_func: Function that returns disturbance dict given (state, t)

        Returns:
            t, x, kappas: Time, state history, and computed slip ratios
        """

        def rhs(state, t):
            control = control_func(state, t)
            disturbances = disturbance_func(state, t) if disturbance_func else {}
            return self.dynamics(state, t, control, disturbances)

        x = odeint(rhs, x0, t_span)

        # Compute slip ratios for all time steps
        kappas_history = []
        for i in range(len(t_span)):
            state = x[i]
            vx, vy, r = state[0:3]
            omegas = state[3:7]

            # Get control at this time
            control = control_func(state, t_span[i])
            delta = control["delta"]

            wheel_vels = self.compute_wheel_velocities(vx, vy, r, delta)
            vx_wheels = wheel_vels["vx"]
            kappas = self.compute_slip_ratios(vx_wheels, omegas)
            kappas_history.append(kappas)

        kappas_history = np.array(kappas_history)

        return t_span, x, kappas_history


# Example usage and control strategies
def pid_controller(state, t, target_vx=20.0, drivetrain="RWD"):
    """
    Simple PID controller for longitudinal velocity

    Args:
        state: Current vehicle state
        t: Time
        target_vx: Target longitudinal velocity
        drivetrain: 'RWD', 'FWD', 'AWD', or '4WD'
    """
    vx = state[0]
    error = target_vx - vx

    # Simple P controller
    Kp = 100.0
    T_cmd = Kp * error

    # Distribute torque based on drivetrain configuration
    if drivetrain == "RWD":
        # Rear-wheel drive: 100% rear
        T_motor_cmd = [0, 0, T_cmd / 2, T_cmd / 2]
    elif drivetrain == "FWD":
        # Front-wheel drive: 100% front
        T_motor_cmd = [T_cmd / 2, T_cmd / 2, 0, 0]
    elif drivetrain == "AWD":
        # All-wheel drive: 40% front, 60% rear (typical AWD bias)
        T_motor_cmd = [T_cmd * 0.2, T_cmd * 0.2, T_cmd * 0.3, T_cmd * 0.3]
    elif drivetrain == "4WD":
        # Four-wheel drive: 25% each wheel
        T_motor_cmd = [T_cmd / 4, T_cmd / 4, T_cmd / 4, T_cmd / 4]
    else:
        raise ValueError(f"Unknown drivetrain: {drivetrain}")

    T_brake_cmd = [0, 0, 0, 0]

    # Sinusoidal steering for testing
    delta = 0.05 * np.sin(0.5 * t)

    return {"T_motor_cmd": T_motor_cmd, "T_brake_cmd": T_brake_cmd, "delta": delta}


def pid_controller_rwd(state, t, target_vx=20.0):
    """Rear-wheel drive controller"""
    return pid_controller(state, t, target_vx, drivetrain="RWD")


def pid_controller_fwd(state, t, target_vx=20.0):
    """Front-wheel drive controller"""
    return pid_controller(state, t, target_vx, drivetrain="FWD")


def pid_controller_awd(state, t, target_vx=20.0):
    """All-wheel drive controller"""
    return pid_controller(state, t, target_vx, drivetrain="AWD")


def pid_controller_4wd(state, t, target_vx=20.0):
    """Four-wheel drive controller"""
    return pid_controller(state, t, target_vx, drivetrain="4WD")


def lqr_controller(state, t):
    """Placeholder for LQR controller"""
    # TODO: Implement LQR with linearized model
    return pid_controller(state, t)


def mpc_controller(state, t):
    """Placeholder for MPC controller"""
    # TODO: Implement MPC with CasADi or scipy.optimize
    return pid_controller(state, t)


if __name__ == "__main__":
    # Initialize vehicle
    vehicle = DualTrackVehicleModel()

    # Initial conditions (15 states)
    v0 = 15.0  # m/s
    x0 = np.array(
        [
            v0,
            0,
            0,  # vx, vy, r
            v0 / vehicle.p.Rw,
            v0 / vehicle.p.Rw,
            v0 / vehicle.p.Rw,
            v0 / vehicle.p.Rw,  # wheel speeds
            0,
            0,
            0,
            0,  # motor torques
            0,
            0,
            0,
            0,  # brake torques
        ]
    )

    # Time span
    t = np.linspace(0, 10, 1000)

    # Choose drivetrain configuration
    # Options: 'RWD', 'FWD', 'AWD', '4WD'
    drivetrain = "AWD"  # Change this to test different configurations

    def controller(state, t):
        return pid_controller(state, t, target_vx=20.0, drivetrain=drivetrain)

    # Simulate with PID controller
    print(f"Simulating dual-track vehicle model with {drivetrain} drivetrain...")
    t_sim, x_sim, kappas_sim = vehicle.simulate(x0, t, controller)

    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # Velocities
    axes[0, 0].plot(t_sim, x_sim[:, 0], label="vx")
    axes[0, 0].plot(t_sim, x_sim[:, 1], label="vy")
    axes[0, 0].set_ylabel("Velocity (m/s)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title("Body Velocities")

    # Yaw rate
    axes[0, 1].plot(t_sim, np.rad2deg(x_sim[:, 2]))
    axes[0, 1].set_ylabel("Yaw rate (deg/s)")
    axes[0, 1].grid(True)
    axes[0, 1].set_title("Yaw Rate")

    # Wheel speeds
    axes[1, 0].plot(t_sim, x_sim[:, 3], label="FL")
    axes[1, 0].plot(t_sim, x_sim[:, 4], label="FR")
    axes[1, 0].plot(t_sim, x_sim[:, 5], label="RL")
    axes[1, 0].plot(t_sim, x_sim[:, 6], label="RR")
    axes[1, 0].set_ylabel("Wheel speed (rad/s)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title("Wheel Speeds")

    # Slip ratios (computed, not states)
    axes[1, 1].plot(t_sim, kappas_sim[:, 0], label="FL")
    axes[1, 1].plot(t_sim, kappas_sim[:, 1], label="FR")
    axes[1, 1].plot(t_sim, kappas_sim[:, 2], label="RL")
    axes[1, 1].plot(t_sim, kappas_sim[:, 3], label="RR")
    axes[1, 1].set_ylabel("Slip ratio")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_title("Longitudinal Slip Ratios")

    # Motor torques
    axes[2, 0].plot(t_sim, x_sim[:, 7], label="FL")
    axes[2, 0].plot(t_sim, x_sim[:, 8], label="FR")
    axes[2, 0].plot(t_sim, x_sim[:, 9], label="RL")
    axes[2, 0].plot(t_sim, x_sim[:, 10], label="RR")
    axes[2, 0].set_ylabel("Motor torque (Nm)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    axes[2, 0].set_title("Motor Torques")

    # Brake torques
    axes[2, 1].plot(t_sim, x_sim[:, 11], label="FL")
    axes[2, 1].plot(t_sim, x_sim[:, 12], label="FR")
    axes[2, 1].plot(t_sim, x_sim[:, 13], label="RL")
    axes[2, 1].plot(t_sim, x_sim[:, 14], label="RR")
    axes[2, 1].set_ylabel("Brake torque (Nm)")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    axes[2, 1].set_title("Brake Torques")

    plt.tight_layout()
    print("Simulation complete!")
    plt.show()

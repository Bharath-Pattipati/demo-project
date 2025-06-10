# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# %% Define the enhanced unified bicycle model class
class EnhancedUnifiedBicycleModel:
    """
    Enhanced unified bicycle model with separated steady-state and transient responses.
    Designed for embedded controller deployment with minimal computational overhead.
    """

    def __init__(self, physical_params, tire_model):
        # Physical parameters
        self.M_phys = physical_params["M"]
        self.Izz_phys = physical_params["Izz"]
        self.a = physical_params["a"]
        self.b = physical_params["b"]
        self.L = self.a + self.b

        # Weight distribution (static)
        self.Wf = self.M_phys * 9.81 * self.b / self.L
        self.Wr = self.M_phys * 9.81 * self.a / self.L
        self.Fz_array = np.array([self.Wf, self.Wr])

        # Tire model (flexible: linear, nonlinear, or mixed)
        self.tire_model = tire_model

        # Get baseline stiffness
        baseline_stiffness = self.tire_model.get_linear_stiffness(self.Fz_array)
        self.Cf_baseline, self.Cr_baseline = baseline_stiffness

        # Initialize tuning parameters
        self.reset_tuning_parameters()

        # State tracking for transient response
        self.transient_state = np.array([0.0, 0.0])  # [vy_transient, r_transient]
        self.last_delta_f = 0.0
        self.last_delta_r = 0.0

    def reset_tuning_parameters(self):
        """Reset all tuning parameters to baseline values."""
        self.M_tune = self.M_phys
        self.Izz_tune = self.Izz_phys
        self.Cf_tune = self.Cf_baseline
        self.Cr_tune = self.Cr_baseline
        self.Cf_scale = 1.0
        self.Cr_scale = 1.0

    def set_understeer_gradient(self, Kus_deg_per_g, balance_ratio=None):
        """
        Set understeer gradient in deg/g by adjusting Cf and Cr.
        Uses direct calculation to avoid iterative solvers.
        """
        Kus_target_rad_per_g = Kus_deg_per_g * np.pi / 180

        if balance_ratio is None:
            balance_ratio = self.Cf_baseline / self.Cr_baseline

        # Direct calculation: Kus = (M/L) * (Wf/Cf - Wr/Cr)
        # Solve for Cr given balance_ratio = Cf/Cr
        numerator = self.M_tune * (self.Wf / balance_ratio - self.Wr)
        denominator = Kus_target_rad_per_g * self.L

        if abs(denominator) < 1e-10:
            print("Warning: Cannot set understeer gradient, denominator near zero.")
            return

        Cr_new = numerator / denominator

        # Ensure physical validity
        if Cr_new <= 0:
            print(
                f"Warning: Calculated Cr ({Cr_new:.2f}) is non-positive. Using fallback."
            )
            # Simple fallback: adjust from baseline
            adjustment_factor = max(0.1, min(10.0, abs(Kus_target_rad_per_g) / 0.01))
            Cr_new = self.Cr_baseline / adjustment_factor

        self.Cr_tune = max(Cr_new, 100.0)
        self.Cf_tune = balance_ratio * self.Cr_tune

        # Update scaling factors for tire model
        self.Cf_scale = self.Cf_tune / self.Cf_baseline
        self.Cr_scale = self.Cr_tune / self.Cr_baseline

    def set_response_speed(self, speed_factor):
        """Set response speed by scaling mass and inertia inversely."""
        if speed_factor <= 0:
            raise ValueError("speed_factor must be positive.")
        self.M_tune = self.M_phys / speed_factor
        self.Izz_tune = self.Izz_phys / speed_factor

    def set_stiffness_ratio(self, cf_cr_ratio):
        """Set Cf/Cr ratio while maintaining current understeer gradient."""
        current_kus = self.get_understeer_gradient()
        self.set_understeer_gradient(current_kus, balance_ratio=cf_cr_ratio)

    def set_balance_delta(self, balance_delta):
        """Adjust balance by adding/subtracting from Cf and Cr."""
        self.Cf_tune += balance_delta
        self.Cr_tune -= balance_delta

        # Ensure positive values
        self.Cf_tune = max(self.Cf_tune, 100.0)
        self.Cr_tune = max(self.Cr_tune, 100.0)

        # Update scaling factors
        self.Cf_scale = self.Cf_tune / self.Cf_baseline
        self.Cr_scale = self.Cr_tune / self.Cr_baseline

    def set_stability_factor(self, stability_factor):
        """Adjust stability by scaling inertia (higher = more stable)."""
        self.Izz_tune = self.Izz_phys * stability_factor

    def calculate_slip_angles(self, state, delta_f, delta_r, vx):
        """
        Calculate slip angles from vehicle state and steering inputs.
        Args:
            state: [vy, r] - lateral velocity, yaw rate
            delta_f: front steer angle (rad)
            delta_r: rear steer angle (rad)
            vx: longitudinal velocity
        Returns:
            [alpha_f, alpha_r] - slip angles (rad)
        """
        vy, r = state

        if abs(vx) < 0.1:
            return np.array([0.0, 0.0])

        alpha_f = delta_f - (vy + self.a * r) / vx
        alpha_r = delta_r - (vy - self.b * r) / vx

        return np.array([alpha_f, alpha_r])

    def calculate_tire_forces(self, slip_angles):
        """
        Calculate tire forces from slip angles using the tire model.
        Returns scaled forces based on tuning parameters.
        """
        # Get forces from tire model
        Fy_forces = self.tire_model.lateral_force(slip_angles, self.Fz_array)

        # Apply scaling factors
        Fyf_scaled = Fy_forces[0] * self.Cf_scale
        Fyr_scaled = Fy_forces[1] * self.Cr_scale

        return np.array([Fyf_scaled, Fyr_scaled])

    def calculate_steady_state_response(self, delta_f, delta_r, vx):
        """
        Calculate steady-state response using numerically robust method.
        Uses analytical solution when possible, falls back to conditioned solve.
        """
        if abs(vx) < 0.1:
            return np.array([0.0, 0.0])  # [vy_ss, r_ss]

        # For bicycle model, we can derive analytical solution directly
        # This avoids matrix operations entirely and is most robust

        # Physical insight: steady-state response for bicycle model
        # vy_ss = (L/vx) * delta_eff * vx^2 / (L + Kus * vx^2)
        # r_ss = vx * delta_eff / (L + Kus * vx^2)
        # where delta_eff accounts for front/rear steering

        # Effective steering input (weighted by tire stiffness)
        Cf_eff = self.Cf_tune
        Cr_eff = self.Cr_tune
        total_stiffness = Cf_eff + Cr_eff

        if total_stiffness < 1e-6:  # Avoid division by zero
            return np.array([0.0, 0.0])

        # Weighted effective steering angle
        delta_eff = (Cf_eff * delta_f + Cr_eff * delta_r) / total_stiffness

        # Understeer gradient (physical units: rad*s^2/m per m/s^2)
        Kus = (self.M_tune / self.L) * (self.Wf / Cf_eff - self.Wr / Cr_eff)

        # Characteristic speed and stability factor
        denominator = self.L + Kus * vx * vx

        # Check for critical speed (denominator → 0)
        if abs(denominator) < 1e-6:
            # Near critical speed - use limiting behavior
            print(f"Warning: Near critical speed at vx={vx:.2f} m/s")
            # Limit response to prevent numerical issues
            max_response = 10.0  # Maximum reasonable lateral velocity
            r_ss = np.sign(delta_eff) * min(
                abs(delta_eff * vx / self.L), max_response / vx
            )
            vy_ss = r_ss * self.L / vx if vx > 0.1 else 0.0
            return np.array([vy_ss, r_ss])

        # Normal case - analytical solution
        r_ss = vx * delta_eff / denominator
        vy_ss = self.L * r_ss / vx if vx > 0.1 else 0.0

        # Sanity check on outputs
        max_reasonable_vy = min(vx * 0.5, 20.0)  # Max 50% of forward speed or 20 m/s
        max_reasonable_r = min(vx / 5.0, 2.0)  # Max reasonable yaw rate

        vy_ss = np.clip(vy_ss, -max_reasonable_vy, max_reasonable_vy)
        r_ss = np.clip(r_ss, -max_reasonable_r, max_reasonable_r)

        return np.array([vy_ss, r_ss])

    def calculate_steady_state_response_robust_fallback(self, delta_f, delta_r, vx):
        """
        Robust fallback using conditioned matrix solve for complex scenarios.
        Only used when analytical solution is not applicable.
        """
        if abs(vx) < 0.1:
            return np.array([0.0, 0.0])

        # Build system matrices with numerical conditioning
        a11 = -(self.Cf_tune + self.Cr_tune) / (self.M_tune * vx)
        a12 = -(self.a * self.Cf_tune - self.b * self.Cr_tune) / (self.M_tune * vx) - vx
        a21 = -(self.a * self.Cf_tune - self.b * self.Cr_tune) / (self.Izz_tune * vx)
        a22 = -(self.a**2 * self.Cf_tune + self.b**2 * self.Cr_tune) / (
            self.Izz_tune * vx
        )

        b11 = self.Cf_tune / self.M_tune
        b12 = self.Cr_tune / self.M_tune
        b21 = self.a * self.Cf_tune / self.Izz_tune
        b22 = -self.b * self.Cr_tune / self.Izz_tune

        # RHS = -B * u
        rhs1 = -(b11 * delta_f + b12 * delta_r)
        rhs2 = -(b21 * delta_f + b22 * delta_r)

        # Robust 2x2 solve using partial pivoting (manually implemented)
        # This is more stable than Cramer's rule

        # Scale equations to improve conditioning
        scale1 = max(abs(a11), abs(a12), abs(rhs1), 1e-12)
        scale2 = max(abs(a21), abs(a22), abs(rhs2), 1e-12)

        a11_s, a12_s, rhs1_s = a11 / scale1, a12 / scale1, rhs1 / scale1
        a21_s, a22_s, rhs2_s = a21 / scale2, a12 / scale2, rhs2 / scale2

        # Choose pivot (largest absolute value in first column)
        if abs(a21_s) > abs(a11_s):
            # Swap rows
            a11_s, a12_s, rhs1_s, a21_s, a22_s, rhs2_s = (
                a21_s,
                a22_s,
                rhs2_s,
                a11_s,
                a12_s,
                rhs1_s,
            )

        # Check for singularity
        if abs(a11_s) < 1e-12:
            print("Warning: Singular steady-state system. Using fallback response.")
            # Fallback: simple proportional response
            return np.array([delta_f * vx * 0.1, delta_f / self.L])

        # Forward elimination
        factor = a21_s / a11_s
        a22_new = a22_s - factor * a12_s
        rhs2_new = rhs2_s - factor * rhs1_s

        # Check for singularity after elimination
        if abs(a22_new) < 1e-12:
            print("Warning: Singular system after elimination. Using fallback.")
            return np.array([delta_f * vx * 0.1, delta_f / self.L])

        # Back substitution
        r_ss = rhs2_new / a22_new
        vy_ss = (rhs1_s - a12_s * r_ss) / a11_s

        # Apply reasonable limits
        max_vy = min(vx * 0.5, 20.0)
        max_r = min(vx / 5.0, 2.0)

        vy_ss = np.clip(vy_ss, -max_vy, max_vy)
        r_ss = np.clip(r_ss, -max_r, max_r)

        return np.array([vy_ss, r_ss])

    def simulation_step(self, x_current, delta_f, delta_r, vx, dt, params):
        """
        Main simulation step using error-based transient dynamics.
        Computational separation: steady-state target vs transient error evolution.

        Args:
            x_current: [vy, r] current total state
            delta_f, delta_r: steering inputs
            vx: longitudinal velocity
            dt: time step
            params: parameter structure (for future flexibility)

        Returns:
            x_next: [vy, r] next total state
        """
        if abs(vx) < 0.1:
            return x_current

        # Step 1: Calculate target steady-state (from Kus calibration only)
        x_ss = self.calculate_steady_state_response(delta_f, delta_r, vx)

        # Step 2: Calculate transient error from target
        x_error = x_current - x_ss

        # Step 3: Evolve transient error with COMPLETE dynamics
        # Use full nonlinear tire model and complete vehicle dynamics
        A_complete = self.get_complete_dynamics_matrix(x_current, delta_f, delta_r, vx)

        # Step 4: Integrate transient dynamics (keeps all coupling!)
        x_error_dot = A_complete @ x_error
        x_error_next = x_error + x_error_dot * dt

        # Step 5: Reconstruct total response
        x_next = x_ss + x_error_next

        return x_next

    def get_complete_dynamics_matrix(self, x_current, delta_f, delta_r, vx):
        """
        Get complete dynamics matrix including nonlinear tire effects.
        This preserves all physical coupling in the transient response.
        """
        if abs(vx) < 0.1:
            return np.zeros((2, 2))

        # Calculate current slip angles and tire forces
        slip_angles = self.calculate_slip_angles(x_current, delta_f, delta_r, vx)

        # For linear approximation around current state, we need tire stiffness
        # This could be linearized Pacejka derivatives or simple linear stiffness
        if hasattr(self.tire_model, "get_local_stiffness"):
            # Use local stiffness from nonlinear tire model
            Cf_local, Cr_local = self.tire_model.get_local_stiffness(
                slip_angles, self.Fz_array
            )
        else:
            # Fallback to tuned linear stiffness
            Cf_local = self.Cf_tune
            Cr_local = self.Cr_tune

        # Apply calibration scaling to local stiffness
        Cf_eff = Cf_local * self.Cf_scale
        Cr_eff = Cr_local * self.Cr_scale

        # Build complete dynamics matrix (includes all coupling)
        A_complete = np.array(
            [
                [
                    -(Cf_eff + Cr_eff) / (self.M_tune * vx),
                    -(self.a * Cf_eff - self.b * Cr_eff) / (self.M_tune * vx) - vx,
                ],
                [
                    -(self.a * Cf_eff - self.b * Cr_eff) / (self.Izz_tune * vx),
                    -(self.a**2 * Cf_eff + self.b**2 * Cr_eff) / (self.Izz_tune * vx),
                ],
            ]
        )

        return A_complete

    def get_total_response(self, delta_f, delta_r, vx):
        """
        Get total vehicle response (steady-state + transient).
        """
        steady_state = self.calculate_steady_state_response(delta_f, delta_r, vx)
        total_state = steady_state + self.transient_state

        # Calculate sideslip angle
        beta = np.arctan(total_state[0] / vx) if vx > 0.1 else 0.0

        return {
            "vy": total_state[0],
            "r": total_state[1],
            "beta": beta,
            "vy_ss": steady_state[0],
            "r_ss": steady_state[1],
            "vy_transient": self.transient_state[0],
            "r_transient": self.transient_state[1],
        }

    def simulate_response(self, delta_f_profile, delta_r_profile, vx, dt=0.01):
        """
        Simulate vehicle response using error-based transient dynamics approach.

        Args:
            delta_f_profile: array of front steering angles over time
            delta_r_profile: array of rear steering angles over time
            vx: longitudinal velocity (constant)
            dt: time step
        Returns:
            Dictionary with time histories of all states
        """
        n_steps = len(delta_f_profile)

        # Initialize output arrays
        results = {
            "time": np.arange(n_steps) * dt,
            "vy": np.zeros(n_steps),
            "r": np.zeros(n_steps),
            "beta": np.zeros(n_steps),
            "vy_ss": np.zeros(n_steps),
            "r_ss": np.zeros(n_steps),
            "vy_error": np.zeros(n_steps),
            "r_error": np.zeros(n_steps),
        }

        # Initialize state - start from rest
        x_current = np.array([0.0, 0.0])  # [vy, r]

        for i in range(n_steps):
            # Calculate steady-state target
            x_ss = self.calculate_steady_state_response(
                delta_f_profile[i], delta_r_profile[i], vx
            )

            # Calculate current error
            x_error = x_current - x_ss

            # Store results
            results["vy"][i] = x_current[0]
            results["r"][i] = x_current[1]
            results["beta"][i] = np.arctan(x_current[0] / vx) if vx > 0.1 else 0.0
            results["vy_ss"][i] = x_ss[0]
            results["r_ss"][i] = x_ss[1]
            results["vy_error"][i] = x_error[0]
            results["r_error"][i] = x_error[1]

            # Evolve to next time step
            if i < n_steps - 1:
                x_current = self.simulation_step(
                    x_current, delta_f_profile[i], delta_r_profile[i], vx, dt, {}
                )

        return results

    # Update the get_total_response method to work with the new architecture
    def get_current_response(self, x_current, delta_f, delta_r, vx):
        """
        Get current vehicle response decomposed into steady-state and error components.
        """
        steady_state = self.calculate_steady_state_response(delta_f, delta_r, vx)
        error_state = x_current - steady_state

        # Calculate sideslip angle
        beta = np.arctan(x_current[0] / vx) if vx > 0.1 else 0.0

        return {
            "vy": x_current[0],
            "r": x_current[1],
            "beta": beta,
            "vy_ss": steady_state[0],
            "r_ss": steady_state[1],
            "vy_error": error_state[0],
            "r_error": error_state[1],
        }

    def get_understeer_gradient(self):
        """Get current understeer gradient in deg/g."""
        if self.Cf_tune == 0 or self.Cr_tune == 0:
            return np.inf

        Kus_rad_per_g = (self.M_tune / self.L) * (
            self.Wf / self.Cf_tune - self.Wr / self.Cr_tune
        )
        return Kus_rad_per_g * 180 / np.pi

    # Enhanced calibration interface for steady-state handling dynamics
    def set_kus_target(self, Kus_deg_per_g):
        """Primary calibration: Set understeer gradient (affects steady-state only)."""
        self.set_understeer_gradient(Kus_deg_per_g)

    def set_cf_cr_difference(self, cf_minus_cr):
        """Calibration: Set Cf - Cr difference while maintaining Kus."""
        current_kus = self.get_understeer_gradient()
        current_sum = self.Cf_tune + self.Cr_tune

        # Solve system: Cf + Cr = current_sum, Cf - Cr = cf_minus_cr
        self.Cf_tune = (current_sum + cf_minus_cr) / 2
        self.Cr_tune = (current_sum - cf_minus_cr) / 2

        # Ensure positive values
        self.Cf_tune = max(self.Cf_tune, 100.0)
        self.Cr_tune = max(self.Cr_tune, 100.0)

        # Update scaling factors
        self.Cf_scale = self.Cf_tune / self.Cf_baseline
        self.Cr_scale = self.Cr_tune / self.Cr_baseline

        # Re-establish Kus (may have changed due to clipping)
        self.set_understeer_gradient(current_kus)

    def set_cf_cr_ratio(self, cf_over_cr_ratio):
        """Calibration: Set Cf/Cr ratio while maintaining Kus."""
        current_kus = self.get_understeer_gradient()
        self.set_understeer_gradient(current_kus, balance_ratio=cf_over_cr_ratio)

    # Enhanced calibration interface for transient/stability dynamics
    def set_mass_tune(self, mass_factor):
        """Calibration: Set effective mass (affects transient response speed)."""
        self.M_tune = self.M_phys * mass_factor

    def set_inertia_tune(self, inertia_factor):
        """Calibration: Set effective inertia (affects stability and yaw response)."""
        self.Izz_tune = self.Izz_phys * inertia_factor

    def set_transient_speed(self, speed_factor):
        """Calibration: Set overall transient response speed (scales both M and Izz)."""
        self.set_response_speed(speed_factor)

    def get_calibration_summary(self):
        """Get summary of all calibration parameters for tuning interface."""
        return {
            # Steady-state handling calibrations
            "Kus_deg_per_g": self.get_understeer_gradient(),
            "Cf_tune": self.Cf_tune,
            "Cr_tune": self.Cr_tune,
            "Cf_minus_Cr": self.Cf_tune - self.Cr_tune,
            "Cf_over_Cr": self.Cf_tune / self.Cr_tune if self.Cr_tune > 0 else np.inf,
            # Transient/stability calibrations
            "mass_factor": self.M_tune / self.M_phys,
            "inertia_factor": self.Izz_tune / self.Izz_phys,
            "transient_speed_factor": self.M_phys / self.M_tune,
            # Derived properties
            "steady_state_gain": self.L
            / (self.L + self.get_understeer_gradient() * (180 / np.pi) * 9.81),
            "natural_frequency_approx": np.sqrt(
                (self.Cf_tune + self.Cr_tune) / (self.M_tune * self.L)
            ),
        }


# %% Example usage linear tire model interface
class LinearTireModel:
    """Simple linear tire model for testing."""

    def __init__(self, Cf_base=80000, Cr_base=60000):
        self.Cf_base = Cf_base
        self.Cr_base = Cr_base

    def get_linear_stiffness(self, Fz_array):
        """Return linear cornering stiffness."""
        return np.array([self.Cf_base, self.Cr_base])

    def lateral_force(self, slip_angles, Fz_array):
        """Calculate lateral forces from slip angles."""
        stiffness = self.get_linear_stiffness(Fz_array)
        return stiffness * slip_angles


# %% Enhanced nonlinear tire model with proper local stiffness calculation
class EnhancedPacejkaTireModel:
    """
    Enhanced Pacejka tire model with proper local stiffness calculation
    for use in nonlinear vehicle dynamics.
    """

    def __init__(self, pacejka_params):
        self.params = pacejka_params

        # Pacejka parameters (Magic Formula)
        self.B = pacejka_params.get("B", [10.0, 10.0])  # Stiffness factor [front, rear]
        self.C = pacejka_params.get("C", [1.3, 1.3])  # Shape factor
        self.D = pacejka_params.get("D", [1.0, 1.0])  # Peak factor (will scale with Fz)
        self.E = pacejka_params.get("E", [0.5, 0.5])  # Curvature factor

        # Linear stiffness for small angles (reference)
        self.Cf_linear = pacejka_params.get("Cf_linear", 80000)
        self.Cr_linear = pacejka_params.get("Cr_linear", 60000)

        # Numerical differentiation parameters
        self.h = 0.001  # Small angle increment for numerical derivative

    def get_linear_stiffness(self, Fz_array):
        """Return reference linear cornering stiffness (small angle limit)."""
        return np.array([self.Cf_linear, self.Cr_linear])

    def lateral_force(self, slip_angles, Fz_array):
        """Calculate lateral forces using Magic Formula."""
        forces = np.zeros_like(slip_angles)

        for i, (alpha, Fz) in enumerate(zip(slip_angles, Fz_array)):
            # Scale peak force with normal load
            D_scaled = self.D[i] * Fz * 0.8  # Peak friction ~0.8

            # Magic Formula
            forces[i] = D_scaled * np.sin(
                self.C[i]
                * np.arctan(
                    self.B[i] * alpha
                    - self.E[i] * (self.B[i] * alpha - np.arctan(self.B[i] * alpha))
                )
            )

        return forces

    def get_local_stiffness(self, slip_angles, Fz_array):
        """
        Calculate local (instantaneous) cornering stiffness at current slip angles.
        This is the key method for nonlinear dynamics!
        """
        local_stiffness = np.zeros_like(slip_angles)

        for i, (alpha, Fz) in enumerate(zip(slip_angles, Fz_array)):
            # Calculate derivative dFy/dα at current slip angle
            # Use central difference for better accuracy
            alpha_plus = alpha + self.h
            alpha_minus = alpha - self.h

            Fy_plus = self._single_tire_force(alpha_plus, Fz, i)
            Fy_minus = self._single_tire_force(alpha_minus, Fz, i)

            # Local stiffness = -dFy/dα (negative because Fy opposes α)
            local_stiffness[i] = -(Fy_plus - Fy_minus) / (2 * self.h)

        return local_stiffness

    def _single_tire_force(self, alpha, Fz, tire_index):
        """Calculate force for a single tire (helper for differentiation)."""
        D_scaled = self.D[tire_index] * Fz * 0.8

        return D_scaled * np.sin(
            self.C[tire_index]
            * np.arctan(
                self.B[tire_index] * alpha
                - self.E[tire_index]
                * (self.B[tire_index] * alpha - np.arctan(self.B[tire_index] * alpha))
            )
        )

    def get_stiffness_analysis(self, alpha_range=np.linspace(-0.3, 0.3, 100), Fz=5000):
        """
        Analyze how stiffness varies with slip angle.
        Useful for understanding nonlinear behavior.
        """
        forces = []
        local_stiffness = []

        for alpha in alpha_range:
            # Calculate force and local stiffness
            Fy = self._single_tire_force(alpha, Fz, 0)  # Use front tire params
            Cα_local = self.get_local_stiffness(np.array([alpha]), np.array([Fz]))[0]

            forces.append(Fy)
            local_stiffness.append(Cα_local)

        return {
            "slip_angles_deg": alpha_range * 180 / np.pi,
            "lateral_forces": np.array(forces),
            "local_stiffness": np.array(local_stiffness),
            "linear_stiffness": self.Cf_linear,
        }


# Enhanced vehicle model that properly uses local stiffness
class NonlinearVehicleModel:
    """
    Vehicle model that properly handles nonlinear tire behavior
    by using local stiffness at each operating point.
    """

    def __init__(self, physical_params, tire_model):
        # ... (same initialization as your original model)
        self.M = physical_params["M"]
        self.Izz = physical_params["Izz"]
        self.a = physical_params["a"]
        self.b = physical_params["b"]
        self.L = self.a + self.b

        self.tire_model = tire_model

        # Weight distribution
        self.Wf = self.M * 9.81 * self.b / self.L
        self.Wr = self.M * 9.81 * self.a / self.L
        self.Fz_array = np.array([self.Wf, self.Wr])

    def get_dynamics_matrix_nonlinear(self, state, delta_f, delta_r, vx):
        """
        Get dynamics matrix using local tire stiffness at current operating point.
        This is the correct approach for nonlinear tire models!
        """
        if abs(vx) < 0.1:
            return np.zeros((2, 2))

        # Calculate current slip angles
        slip_angles = self.calculate_slip_angles(state, delta_f, delta_r, vx)

        # Get local stiffness at current operating point
        local_stiffness = self.tire_model.get_local_stiffness(
            slip_angles, self.Fz_array
        )
        Cf_local, Cr_local = local_stiffness

        # Build dynamics matrix with local stiffness
        A = np.array(
            [
                [
                    -(Cf_local + Cr_local) / (self.M * vx),
                    -(self.a * Cf_local - self.b * Cr_local) / (self.M * vx) - vx,
                ],
                [
                    -(self.a * Cf_local - self.b * Cr_local) / (self.Izz * vx),
                    -(self.a**2 * Cf_local + self.b**2 * Cr_local) / (self.Izz * vx),
                ],
            ]
        )

        return A

    def calculate_slip_angles(self, state, delta_f, delta_r, vx):
        """Calculate slip angles from vehicle state."""
        vy, r = state

        if abs(vx) < 0.1:
            return np.array([0.0, 0.0])

        alpha_f = delta_f - (vy + self.a * r) / vx
        alpha_r = delta_r - (vy - self.b * r) / vx

        return np.array([alpha_f, alpha_r])


# Example usage and validation
def demonstrate_nonlinear_effects():
    """Demonstrate the difference between linear and nonlinear approaches."""

    # Setup tire model
    pacejka_params = {
        "B": [12.0, 10.0],  # Front/rear stiffness factors
        "C": [1.4, 1.3],  # Shape factors
        "D": [1.0, 1.0],  # Peak factors
        "E": [0.3, 0.4],  # Curvature factors
        "Cf_linear": 100000,  # Linear reference
        "Cr_linear": 80000,
    }

    tire_model = EnhancedPacejkaTireModel(pacejka_params)

    # Analyze stiffness variation
    analysis = tire_model.get_stiffness_analysis()

    print("Nonlinear Tire Behavior Analysis:")
    print("=================================")

    # Find where stiffness drops to 50% of linear value
    half_stiff_idx = np.where(
        analysis["local_stiffness"] < analysis["linear_stiffness"] * 0.5
    )[0]
    if len(half_stiff_idx) > 0:
        critical_angle = analysis["slip_angles_deg"][half_stiff_idx[0]]
        print(f"Stiffness drops to 50% of linear at: {critical_angle:.1f}°")

    # Find peak force angle
    peak_idx = np.argmax(np.abs(analysis["lateral_forces"]))
    peak_angle = analysis["slip_angles_deg"][peak_idx]
    peak_stiffness = analysis["local_stiffness"][peak_idx]

    print(f"Peak force occurs at: {peak_angle:.1f}°")
    print(f"Local stiffness at peak: {peak_stiffness:.0f} N/rad")
    print(f"Linear stiffness: {analysis['linear_stiffness']:.0f} N/rad")
    print(
        f"Stiffness ratio at peak: {peak_stiffness / analysis['linear_stiffness']:.2f}"
    )

    return analysis


# Alternative: Piecewise linear approximation for embedded systems
class PiecewiseLinearTireModel:
    """
    Computationally efficient piecewise linear tire model.
    Good compromise between accuracy and computational cost.
    """

    def __init__(self, breakpoints_deg, stiffness_values):
        """
        breakpoints_deg: [0, 2, 5, 10, 15] - slip angle breakpoints in degrees
        stiffness_values: corresponding stiffness values at each breakpoint
        """
        self.breakpoints_rad = np.array(breakpoints_deg) * np.pi / 180
        self.stiffness_front = np.array(stiffness_values["front"])
        self.stiffness_rear = np.array(stiffness_values["rear"])

    def get_local_stiffness(self, slip_angles, Fz_array):
        """Get stiffness using piecewise linear interpolation."""
        local_stiffness = np.zeros_like(slip_angles)
        stiffness_arrays = [self.stiffness_front, self.stiffness_rear]

        for i, alpha in enumerate(slip_angles):
            alpha_abs = abs(alpha)
            stiffness_table = stiffness_arrays[i]

            # Find appropriate stiffness by interpolation
            local_stiffness[i] = np.interp(
                alpha_abs, self.breakpoints_rad, stiffness_table
            )

        return local_stiffness


# Example piecewise setup
piecewise_example = {
    "breakpoints_deg": [0, 2, 5, 8, 12, 20],
    "stiffness_values": {
        "front": [100000, 95000, 70000, 40000, 20000, 5000],  # Decreasing stiffness
        "rear": [80000, 76000, 56000, 32000, 16000, 4000],
    },
}

# Run the demonstration
if __name__ == "__main__":
    analysis = demonstrate_nonlinear_effects()


# %% Tire Force Transformation Comparison
class TireForceTransformation:
    """
    Demonstrates different methods for transforming tire forces to vehicle CG.
    Compares rotation matrix approach vs. direct moment arm calculation.
    """

    def __init__(self, vehicle_params):
        self.a = vehicle_params["a"]  # Distance from CG to front axle
        self.b = vehicle_params["b"]  # Distance from CG to rear axle
        self.L = self.a + self.b  # Wheelbase
        self.track_width = vehicle_params.get("track_width", 1.6)  # For 4-wheel model

    def bicycle_model_moment_arms(self, Fyf, Fyr, delta_f=0, delta_r=0):
        """
        Current approach in your code: Direct moment arm calculation for bicycle model.
        This is the standard and most efficient method for bicycle models.
        """
        # Lateral force (already aligned with vehicle Y-axis in bicycle model)
        Fy_total = Fyf + Fyr

        # Yaw moment about CG using moment arms
        # Positive yaw moment = left turn (counter-clockwise)
        Mz = self.a * Fyf * np.cos(delta_f) - self.b * Fyr * np.cos(delta_r)

        # Add steering moments if significant
        if abs(delta_f) > 0.1 or abs(delta_r) > 0.1:
            # Longitudinal components of steered forces create additional yaw moment
            Mz += (
                self.a * Fyf * np.sin(delta_f) * 0
            )  # Usually negligible for small angles

        return {"Fy_total": Fy_total, "Mz_total": Mz, "method": "moment_arms"}

    def rotation_matrix_approach(self, tire_forces_local, tire_positions, tire_angles):
        """
        Rotation matrix approach for transforming tire forces to CG.
        More general but computationally heavier - useful for 4-wheel models.

        Args:
            tire_forces_local: [(Fx_tire, Fy_tire), ...] forces in tire coordinate system
            tire_positions: [(x_tire, y_tire), ...] tire positions relative to CG
            tire_angles: [delta_1, delta_2, ...] tire steering angles
        """
        Fx_total = 0
        Fy_total = 0
        Mz_total = 0

        for i, ((Fx_local, Fy_local), (x_tire, y_tire), delta) in enumerate(
            zip(tire_forces_local, tire_positions, tire_angles)
        ):
            # Rotation matrix from tire to vehicle coordinates
            # Positive delta = left turn (counter-clockwise rotation)
            cos_delta = np.cos(delta)
            sin_delta = np.sin(delta)

            R = np.array(
                [
                    [cos_delta, -sin_delta],  # Transform to vehicle X (longitudinal)
                    [sin_delta, cos_delta],  # Transform to vehicle Y (lateral)
                ]
            )

            # Transform forces to vehicle coordinate system
            F_local = np.array([Fx_local, Fy_local])
            F_vehicle = R @ F_local

            Fx_vehicle = F_vehicle[0]
            Fy_vehicle = F_vehicle[1]

            # Accumulate forces
            Fx_total += Fx_vehicle
            Fy_total += Fy_vehicle

            # Calculate moment about CG
            # Mz = r × F (cross product in 2D)
            Mz_tire = x_tire * Fy_vehicle - y_tire * Fx_vehicle
            Mz_total += Mz_tire

        return {
            "Fx_total": Fx_total,
            "Fy_total": Fy_total,
            "Mz_total": Mz_total,
            "method": "rotation_matrix",
        }

    def compare_methods_bicycle(self, Fyf, Fyr, delta_f, delta_r):
        """Compare both methods for bicycle model case."""

        # Method 1: Direct moment arms (your current approach)
        result1 = self.bicycle_model_moment_arms(Fyf, Fyr, delta_f, delta_r)

        # Method 2: Rotation matrix approach
        tire_forces_local = [
            (0, Fyf),  # Front tire: only lateral force in tire coordinates
            (0, Fyr),  # Rear tire: only lateral force in tire coordinates
        ]
        tire_positions = [
            (self.a, 0),  # Front axle position
            (-self.b, 0),  # Rear axle position
        ]
        tire_angles = [delta_f, delta_r]

        result2 = self.rotation_matrix_approach(
            tire_forces_local, tire_positions, tire_angles
        )

        return result1, result2

    def four_wheel_example(self, corner_forces, steer_angles):
        """
        Example for 4-wheel vehicle where rotation matrix approach is more natural.

        Args:
            corner_forces: [(Fx_FL, Fy_FL), (Fx_FR, Fy_FR), (Fx_RL, Fy_RL), (Fx_RR, Fy_RR)]
            steer_angles: [delta_FL, delta_FR, delta_RL, delta_RR]
        """
        # Tire positions relative to CG
        tire_positions = [
            (self.a, self.track_width / 2),  # Front left
            (self.a, -self.track_width / 2),  # Front right
            (-self.b, self.track_width / 2),  # Rear left
            (-self.b, -self.track_width / 2),  # Rear right
        ]

        return self.rotation_matrix_approach(
            corner_forces, tire_positions, steer_angles
        )

    def visualize_force_transformation(
        self, Fyf=5000, Fyr=3000, delta_f=0.1, delta_r=0
    ):
        """Visualize the force transformation process."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Vehicle layout with forces
        ax1.set_aspect("equal")

        # Draw vehicle outline
        vehicle_width = 0.8
        vehicle_outline = np.array(
            [
                [-self.b, -vehicle_width / 2],
                [self.a, -vehicle_width / 2],
                [self.a, vehicle_width / 2],
                [-self.b, vehicle_width / 2],
                [-self.b, -vehicle_width / 2],
            ]
        )
        ax1.plot(vehicle_outline[:, 0], vehicle_outline[:, 1], "k-", linewidth=2)

        # Draw CG
        ax1.plot(0, 0, "ro", markersize=8, label="CG")

        # Draw tire forces
        scale = 1e-4  # Scale factor for force visualization

        # Front tire force
        Fx_f = -Fyf * np.sin(delta_f)  # Small angle: Fx ≈ -Fy * δ
        Fy_f = Fyf * np.cos(delta_f)  # Small angle: Fy ≈ Fy * 1

        ax1.arrow(
            self.a,
            0,
            Fx_f * scale,
            Fy_f * scale,
            head_width=0.1,
            head_length=0.1,
            fc="blue",
            ec="blue",
            label=f"Front: {Fyf:.0f}N",
        )

        # Rear tire force
        Fx_r = -Fyr * np.sin(delta_r)
        Fy_r = Fyr * np.cos(delta_r)

        ax1.arrow(
            -self.b,
            0,
            Fx_r * scale,
            Fy_r * scale,
            head_width=0.1,
            head_length=0.1,
            fc="red",
            ec="red",
            label=f"Rear: {Fyr:.0f}N",
        )

        # Show moment arms
        ax1.plot(
            [0, self.a], [0, 0], "g--", alpha=0.5, label=f"Moment arm: {self.a:.1f}m"
        )
        ax1.plot([0, -self.b], [0, 0], "g--", alpha=0.5)

        ax1.set_xlabel("Longitudinal (m)")
        ax1.set_ylabel("Lateral (m)")
        ax1.set_title("Vehicle Forces and Moment Arms")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot: Method comparison
        result1, result2 = self.compare_methods_bicycle(Fyf, Fyr, delta_f, delta_r)

        methods = ["Moment Arms", "Rotation Matrix"]
        fy_values = [result1["Fy_total"], result2["Fy_total"]]
        mz_values = [result1["Mz_total"], result2["Mz_total"]]

        x = np.arange(len(methods))
        width = 0.35

        ax2.bar(x - width / 2, fy_values, width, label="Lateral Force (N)", alpha=0.7)
        ax2.bar(x + width / 2, mz_values, width, label="Yaw Moment (Nm)", alpha=0.7)

        ax2.set_xlabel("Method")
        ax2.set_ylabel("Force/Moment")
        ax2.set_title("Method Comparison")
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def demonstrate_sign_conventions(self):
        """Demonstrate sign conventions for left-turn positive."""

        print("Sign Convention Analysis (Left-turn positive):")
        print("=" * 50)

        # Test case: Left turn with front steering
        Fyf = 5000  # Positive lateral force (leftward)
        Fyr = -2000  # Negative lateral force (rightward, for stability)
        delta_f = 0.05  # Small left steer angle (positive)

        result1, result2 = self.compare_methods_bicycle(Fyf, Fyr, delta_f, 0)

        print(f"Front tire force: {Fyf:+.0f} N (leftward)")
        print(f"Rear tire force: {Fyr:+.0f} N (rightward)")
        print(f"Front steer angle: {delta_f:+.3f} rad (left)")
        print()

        print("Results:")
        print(f"Total lateral force: {result1['Fy_total']:+.0f} N")
        print(f"Yaw moment (moment arms): {result1['Mz_total']:+.0f} Nm")
        print(f"Yaw moment (rotation matrix): {result2['Mz_total']:+.0f} Nm")
        print()

        if result1["Mz_total"] > 0:
            print("✓ Positive yaw moment → Left turn (counter-clockwise)")
        else:
            print("✗ Negative yaw moment → Right turn (clockwise)")

        return result1, result2


# Example usage and validation
if __name__ == "__main__":
    # Vehicle parameters (similar to your model)
    vehicle_params = {
        "a": 1.4,  # Front axle to CG
        "b": 1.6,  # Rear axle to CG
        "track_width": 1.6,
    }

    transformer = TireForceTransformation(vehicle_params)

    # Demonstrate sign conventions
    transformer.demonstrate_sign_conventions()

    # Create visualization
    fig = transformer.visualize_force_transformation()
    plt.show()

    # 4-wheel example
    print("\n4-Wheel Vehicle Example:")
    print("=" * 30)

    corner_forces = [
        (0, 2500),  # FL: Lateral force only
        (0, 2500),  # FR: Lateral force only
        (0, -1000),  # RL: Stabilizing force
        (0, -1000),  # RR: Stabilizing force
    ]

    steer_angles = [0.05, 0.05, 0, 0]  # Front wheels steered left

    result_4w = transformer.four_wheel_example(corner_forces, steer_angles)
    print(f"Total lateral force: {result_4w['Fy_total']:+.0f} N")
    print(f"Total yaw moment: {result_4w['Mz_total']:+.0f} Nm")

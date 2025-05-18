# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
# import matplotlib.cm as cm
# from mpl_toolkits.mplot3d import Axes3D


# %% Vehicle Model Class
class VehicleModel:
    """
    Base class for vehicle models. Implement your specific vehicle model by inheriting from this class.
    """

    def __init__(self, params=None):
        """
        Initialize vehicle model with parameters.

        Args:
            params (dict): Dictionary of vehicle parameters
        """
        self.params = params if params is not None else {}

    def state_derivatives(self, t, state):
        """
        Calculate the derivatives of the state variables.

        Args:
            t (float): Time
            state (np.ndarray): State vector

        Returns:
            np.ndarray: Derivatives of state vector
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_state_names(self):
        """
        Get the names of state variables.

        Returns:
            list: List of state variable names
        """
        raise NotImplementedError("Subclasses must implement this method.")


# %% Dual Track Bicycle Model Class
class DualTrackBicycleModel(VehicleModel):
    """
    Dual track nonlinear bicycle model.
    """

    def __init__(self, params=None):
        """
        Initialize the dual track bicycle model with default parameters if none are provided.

        Args:
            params (dict, optional): Dictionary of vehicle parameters
        """
        default_params = {
            "m": 1500.0,  # Vehicle mass (kg)
            "Iz": 3000.0,  # Yaw moment of inertia (kg.m^2)
            "lf": 1.2,  # Distance from CG to front axle (m)
            "lr": 1.6,  # Distance from CG to rear axle (m)
            "Cf": 80000.0,  # Front cornering stiffness (N/rad)
            "Cr": 80000.0,  # Rear cornering stiffness (N/rad)
            "mu": 1.0,  # Road friction coefficient
            "g": 9.81,  # Gravitational acceleration (m/s^2)
            "width": 1.8,  # Vehicle width (m)
        }

        # Update default parameters with provided parameters
        if params is not None:
            default_params.update(params)

        super().__init__(default_params)

    def state_derivatives(self, t, state):
        """
        Calculate the derivatives of the state variables for the dual track bicycle model.

        State vector:
            [vx, vy, r]
            vx: longitudinal velocity (m/s)
            vy: lateral velocity (m/s)
            r: yaw rate (rad/s)

        Args:
            t (float): Time
            state (np.ndarray): State vector [vx, vy, r]

        Returns:
            np.ndarray: Derivatives of state vector [dvx/dt, dvy/dt, dr/dt]
        """
        vx, vy, r = state

        # Extract parameters
        m = self.params["m"]
        Iz = self.params["Iz"]
        lf = self.params["lf"]
        lr = self.params["lr"]
        Cf = self.params["Cf"]
        Cr = self.params["Cr"]

        # For simplicity, assuming constant longitudinal velocity and zero steering input
        # In a real model, these would be inputs or controlled variables
        delta = 0.0  # Steering angle (rad)
        Fx = 0.0  # Longitudinal force (N)

        # Slip angles
        alpha_f = delta - np.arctan2((vy + lf * r), vx) if vx > 0.1 else 0
        alpha_r = -np.arctan2((vy - lr * r), vx) if vx > 0.1 else 0

        # Forces
        Fyf = Cf * alpha_f
        Fyr = Cr * alpha_r

        # State derivatives
        dvx = Fx / m + vy * r
        dvy = (Fyf + Fyr) / m - vx * r
        dr = (lf * Fyf - lr * Fyr) / Iz

        return np.array([dvx, dvy, dr])

    def get_state_names(self):
        """
        Get the names of state variables.

        Returns:
            list: List of state variable names
        """
        return ["vx", "vy", "r"]

    def get_beta(self, vx, vy):
        """
        Calculate side slip angle (beta).

        Args:
            vx (float or np.ndarray): Longitudinal velocity
            vy (float or np.ndarray): Lateral velocity

        Returns:
            float or np.ndarray: Side slip angle (rad)
        """
        return np.arctan2(vy, vx)


# %% Phase Portrait Analyzer Class
class PhasePortraitAnalyzer:
    """
    Class for analyzing phase portraits of dynamic systems.
    """

    def __init__(self, model):
        """
        Initialize the analyzer with a vehicle model.

        Args:
            model (VehicleModel): Vehicle model to analyze
        """
        self.model = model

    def compute_vector_field(self, x_var, y_var, grid_size=20, fixed_params=None):
        """
        Compute vector field for phase portrait.

        Args:
            x_var (dict): Dictionary with 'name', 'range', and 'index' for x-axis variable
            y_var (dict): Dictionary with 'name', 'range', and 'index' for y-axis variable
            grid_size (int): Number of points in each dimension of the grid
            fixed_params (dict): Dictionary of fixed parameters for other state variables

        Returns:
            tuple: Meshgrid and vector components
        """
        if fixed_params is None:
            fixed_params = {}

        # Create grid
        x = np.linspace(x_var["range"][0], x_var["range"][1], grid_size)
        y = np.linspace(y_var["range"][0], y_var["range"][1], grid_size)
        X, Y = np.meshgrid(x, y)

        # Initialize vector components
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # Number of state variables
        state_names = self.model.get_state_names()
        n_states = len(state_names)

        # Compute vector field
        for i in range(grid_size):
            for j in range(grid_size):
                # Create state vector with fixed parameters
                state = np.zeros(n_states)
                for k, name in enumerate(state_names):
                    if k == x_var["index"]:
                        state[k] = X[i, j]
                    elif k == y_var["index"]:
                        state[k] = Y[i, j]
                    else:
                        state[k] = fixed_params.get(name, 0.0)

                # Compute derivatives
                derivatives = self.model.state_derivatives(0, state)

                # Extract vector components
                U[i, j] = derivatives[x_var["index"]]
                V[i, j] = derivatives[y_var["index"]]

        return X, Y, U, V

    def compute_phase_portrait(
        self,
        x_var,
        y_var,
        grid_size=20,
        fixed_params=None,
        streamplot=True,
        quiver=False,
        normalize=True,
    ):
        """
        Compute and plot phase portrait.

        Args:
            x_var (dict): Dictionary with 'name', 'range', and 'index' for x-axis variable
            y_var (dict): Dictionary with 'name', 'range', and 'index' for y-axis variable
            grid_size (int): Number of points in each dimension of the grid
            fixed_params (dict): Dictionary of fixed parameters for other state variables
            streamplot (bool): Whether to draw streamlines
            quiver (bool): Whether to draw quiver plot
            normalize (bool): Whether to normalize vector lengths

        Returns:
            tuple: Figure and axes objects
        """
        if fixed_params is None:
            fixed_params = {}

        # Compute vector field
        X, Y, U, V = self.compute_vector_field(x_var, y_var, grid_size, fixed_params)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize vectors if requested
        if normalize:
            # Avoid division by zero
            norm = np.sqrt(U**2 + V**2)
            mask = norm > 1e-10
            U_norm = np.zeros_like(U)
            V_norm = np.zeros_like(V)
            U_norm[mask] = U[mask] / norm[mask]
            V_norm[mask] = V[mask] / norm[mask]

            # Original magnitudes for color mapping
            magnitudes = norm

            # Use normalized vectors for plotting
            U_plot, V_plot = U_norm, V_norm
        else:
            magnitudes = np.sqrt(U**2 + V**2)
            U_plot, V_plot = U, V

        # Draw streamlines if requested
        if streamplot:
            strm = ax.streamplot(
                X,
                Y,
                U_plot,
                V_plot,
                color=magnitudes,
                cmap="viridis",
                linewidth=1.5,
                density=1.5,
                arrowsize=1.5,
            )
            plt.colorbar(strm.lines, ax=ax, label="Vector magnitude")

        # Draw quiver plot if requested
        if quiver:
            q = ax.quiver(
                X, Y, U_plot, V_plot, magnitudes, cmap="viridis", scale=25, width=0.002
            )
            plt.colorbar(q, ax=ax, label="Vector magnitude")

        # Set labels and title
        ax.set_xlabel(f"{x_var['name']}")
        ax.set_ylabel(f"{y_var['name']}")
        title = f"Phase Portrait: {y_var['name']} vs {x_var['name']}"
        if fixed_params:
            fixed_str = ", ".join([f"{k}={v:.1f}" for k, v in fixed_params.items()])
            title += f" (Fixed: {fixed_str})"
        ax.set_title(title)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.6)

        # Set aspect ratio
        ax.set_aspect("auto")

        return fig, ax

    def compute_trajectories(
        self, x_var, y_var, initial_conditions, t_span, fixed_params=None
    ):
        """
        Compute trajectories from given initial conditions.

        Args:
            x_var (dict): Dictionary with 'name', 'range', and 'index' for x-axis variable
            y_var (dict): Dictionary with 'name', 'range', and 'index' for y-axis variable
            initial_conditions (list): List of initial state vectors
            t_span (tuple): Time span (t_start, t_end)
            fixed_params (dict): Dictionary of fixed parameters for other state variables

        Returns:
            list: List of trajectory solutions
        """
        if fixed_params is None:
            fixed_params = {}

        state_names = self.model.get_state_names()
        n_states = len(state_names)
        trajectories = []

        # Solve for each initial condition
        for init_cond in initial_conditions:
            # Create initial state vector with fixed parameters
            init_state = np.zeros(n_states)
            for k, name in enumerate(state_names):
                if k in [x_var["index"], y_var["index"]]:
                    idx = x_var["index"] if name == x_var["name"] else y_var["index"]
                    init_state[k] = init_cond[0 if idx == x_var["index"] else 1]
                else:
                    init_state[k] = fixed_params.get(name, 0.0)

            # Define ODE function for solve_ivp
            def ode_func(t, state):
                return self.model.state_derivatives(t, state)

            # Solve ODE
            sol = solve_ivp(
                ode_func,
                t_span,
                init_state,
                method="RK45",
                dense_output=True,
                rtol=1e-4,
                atol=1e-7,
            )

            trajectories.append(sol)

        return trajectories

    def plot_trajectories(self, ax, trajectories, x_var, y_var, colors=None):
        """
        Plot trajectories on given axes.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on
            trajectories (list): List of trajectory solutions
            x_var (dict): Dictionary with 'name', 'range', and 'index' for x-axis variable
            y_var (dict): Dictionary with 'name', 'range', and 'index' for y-axis variable
            colors (list, optional): List of colors for trajectories

        Returns:
            list: List of plotted lines
        """
        if colors is None:
            colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))

        lines = []

        for i, traj in enumerate(trajectories):
            x = traj.y[x_var["index"], :]
            y = traj.y[y_var["index"], :]

            # Plot trajectory
            (line,) = ax.plot(
                x, y, "-", color=colors[i], linewidth=2, label=f"Trajectory {i + 1}"
            )

            # Mark initial point
            ax.plot(x[0], y[0], "o", color=colors[i], markersize=6)

            # Mark direction with an arrow near the end
            if len(x) > 10:
                idx = len(x) - 10
                dx = x[idx + 5] - x[idx]
                dy = y[idx + 5] - y[idx]
                ax.arrow(
                    x[idx],
                    y[idx],
                    dx,
                    dy,
                    head_width=0.05,
                    head_length=0.1,
                    fc=colors[i],
                    ec=colors[i],
                )

            lines.append(line)

        return lines

    def analyze_stability(self, fixed_params=None, grid_size=10):
        """
        Analyze stability of equilibrium points.

        Args:
            fixed_params (dict): Dictionary of fixed parameters for other state variables
            grid_size (int): Grid size for searching equilibrium points

        Returns:
            tuple: List of equilibrium points and their stability
        """
        if fixed_params is None:
            fixed_params = {}

        state_names = self.model.get_state_names()
        n_states = len(state_names)

        # Define ranges for each state variable
        ranges = {
            "vx": (1.0, 30.0),  # Longitudinal velocity
            "vy": (-10.0, 10.0),  # Lateral velocity
            "r": (-1.0, 1.0),  # Yaw rate
        }

        # Create grid for each unfixed state variable
        unfixed_states = []
        for i, name in enumerate(state_names):
            if name not in fixed_params:
                unfixed_states.append(
                    {
                        "name": name,
                        "index": i,
                        "range": ranges.get(name, (-10, 10)),
                        "values": np.linspace(
                            ranges.get(name, (-10, 10))[0],
                            ranges.get(name, (-10, 10))[1],
                            grid_size,
                        ),
                    }
                )

        # Initialize list for equilibrium points
        equilibrium_points = []
        stability = []

        # Check if we have 2 unfixed variables for standard phase plane analysis
        if len(unfixed_states) == 2:
            x_var, y_var = unfixed_states

            # Compute vector field
            X, Y, U, V = self.compute_vector_field(
                x_var, y_var, grid_size, fixed_params
            )

            # Find potential equilibrium points (where derivatives are close to zero)
            for i in range(grid_size):
                for j in range(grid_size):
                    if abs(U[i, j]) < 1e-2 and abs(V[i, j]) < 1e-2:
                        # Create state vector with fixed parameters
                        state = np.zeros(n_states)
                        for k, name in enumerate(state_names):
                            if k == x_var["index"]:
                                state[k] = X[i, j]
                            elif k == y_var["index"]:
                                state[k] = Y[i, j]
                            else:
                                state[k] = fixed_params.get(name, 0.0)

                        # Check if it's a true equilibrium
                        derivatives = self.model.state_derivatives(0, state)
                        if np.all(np.abs(derivatives) < 1e-2):
                            equilibrium_points.append(state)

                            # Compute Jacobian numerically for stability analysis
                            J = np.zeros((n_states, n_states))
                            h = 1e-6

                            for m in range(n_states):
                                state_plus = state.copy()
                                state_plus[m] += h
                                derivatives_plus = self.model.state_derivatives(
                                    0, state_plus
                                )

                                J[:, m] = (derivatives_plus - derivatives) / h

                            # Compute eigenvalues
                            eigenvalues = np.linalg.eigvals(J)

                            # Determine stability
                            if np.all(np.real(eigenvalues) < 0):
                                stability.append("Stable")
                            elif np.all(np.real(eigenvalues) > 0):
                                stability.append("Unstable")
                            else:
                                stability.append("Saddle")

        return equilibrium_points, stability

    def yaw_acceleration_analysis(self, r_range, dr_range, vx=20.0, grid_size=20):
        """
        Analyze yaw rate vs yaw acceleration phase plane.

        Args:
            r_range (tuple): Range of yaw rates
            dr_range (tuple): Range of yaw accelerations (for display)
            vx (float): Fixed longitudinal velocity
            grid_size (int): Grid size

        Returns:
            tuple: Figure and axes objects
        """
        # State index mapping
        state_names = self.model.get_state_names()
        r_idx = state_names.index("r")
        vy_idx = state_names.index("vy")
        vx_idx = state_names.index("vx")

        # Create grid
        r_values = np.linspace(r_range[0], r_range[1], grid_size)
        vy_values = np.linspace(-10.0, 10.0, grid_size)  # Range of lateral velocities
        R, VY = np.meshgrid(r_values, vy_values)

        # Initialize matrices for yaw acceleration
        DR = np.zeros_like(R)

        # Compute yaw acceleration for each point
        for i in range(grid_size):
            for j in range(grid_size):
                state = np.zeros(len(state_names))
                state[vx_idx] = vx
                state[vy_idx] = VY[i, j]
                state[r_idx] = R[i, j]

                derivatives = self.model.state_derivatives(0, state)
                DR[i, j] = derivatives[r_idx]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot contour
        levels = np.linspace(dr_range[0], dr_range[1], 20)
        contour = ax.contourf(R, VY, DR, levels=levels, cmap="coolwarm", extend="both")

        # Add contour lines
        contour_lines = ax.contour(R, VY, DR, levels=[0], colors="k", linewidths=2)
        plt.clabel(contour_lines, inline=True, fontsize=10)

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Yaw Acceleration (rad/s²)")

        # Plot zero yaw acceleration line
        # Find the closest points to zero yaw acceleration
        eps = 1e-6
        zero_acc_mask = (DR > -eps) & (DR < eps)
        if np.any(zero_acc_mask):
            ax.plot(
                R[zero_acc_mask], VY[zero_acc_mask], "ko", label="Zero Yaw Acceleration"
            )

        # Set labels and title
        ax.set_xlabel("Yaw Rate (rad/s)")
        ax.set_ylabel("Lateral Velocity (m/s)")
        ax.set_title(f"Yaw Acceleration Analysis (vx = {vx} m/s)")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.6)

        return fig, ax

    def sideslip_yawrate_analysis(self, beta_range, r_range, vx=20.0, grid_size=20):
        """
        Analyze sideslip angle vs yaw rate phase plane.

        Args:
            beta_range (tuple): Range of sideslip angles
            r_range (tuple): Range of yaw rates
            vx (float): Fixed longitudinal velocity
            grid_size (int): Grid size

        Returns:
            tuple: Figure and axes objects
        """
        # State index mapping
        state_names = self.model.get_state_names()
        r_idx = state_names.index("r")
        vy_idx = state_names.index("vy")
        vx_idx = state_names.index("vx")

        # Create grid
        beta_values = np.linspace(beta_range[0], beta_range[1], grid_size)
        r_values = np.linspace(r_range[0], r_range[1], grid_size)
        BETA, R = np.meshgrid(beta_values, r_values)

        # Convert beta to vy
        VY = vx * np.tan(BETA)

        # Initialize derivative matrices
        DBETA = np.zeros_like(BETA)
        DR = np.zeros_like(R)

        # Compute derivatives for each point
        for i in range(grid_size):
            for j in range(grid_size):
                state = np.zeros(len(state_names))
                state[vx_idx] = vx
                state[vy_idx] = VY[i, j]
                state[r_idx] = R[i, j]

                derivatives = self.model.state_derivatives(0, state)

                # Convert dvy to dbeta
                # beta = arctan(vy/vx)
                # dbeta/dt = (1/(1 + (vy/vx)^2)) * (dvx*vy - dvy*vx)/(vx^2)
                # For constant vx: dbeta/dt = dvy/(vx*(1 + (vy/vx)^2))
                # Simplifies to: dbeta/dt = dvy/(vx + vy^2/vx)
                DBETA[i, j] = derivatives[vy_idx] / (vx + VY[i, j] ** 2 / vx)
                DR[i, j] = derivatives[r_idx]

        # Compute vector field magnitude for color mapping
        magnitude = np.sqrt(DBETA**2 + DR**2)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot streamlines
        strm = ax.streamplot(
            BETA,
            R,
            DBETA,
            DR,
            color=magnitude,
            cmap="viridis",
            linewidth=1.5,
            density=1.5,
            arrowsize=1.5,
        )

        # Add colorbar
        cbar = plt.colorbar(strm.lines, ax=ax)
        cbar.set_label("Vector Magnitude")

        # Set labels and title
        ax.set_xlabel("Sideslip Angle β (rad)")
        ax.set_ylabel("Yaw Rate (rad/s)")
        ax.set_title(f"Sideslip vs Yaw Rate Phase Plane (vx = {vx} m/s)")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.6)

        return fig, ax

    def comprehensive_analysis(
        self, vx=20.0, r_range=(-1.0, 1.0), vy_range=(-10.0, 10.0), grid_size=20
    ):
        """
        Perform comprehensive phase plane analysis with focus on equilibrium points.

        Args:
        vx (float): Fixed longitudinal velocity (m/s)
        r_range (tuple): Range of yaw rates (rad/s)
        vy_range (tuple): Range of lateral velocities (m/s)
        grid_size (int): Resolution of the analysis grid

        Returns:
        tuple: Figure object, axes, equilibrium points, types, and eigenvalues
        """

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(2, 2, figure=fig)

        # State index mapping
        state_names = self.model.get_state_names()
        r_idx = state_names.index("r")
        vy_idx = state_names.index("vy")
        vx_idx = state_names.index("vx")

        # Define variables for phase portrait
        x_var = {"name": "r", "range": r_range, "index": r_idx}
        y_var = {"name": "vy", "range": vy_range, "index": vy_idx}
        fixed_params = {"vx": vx}

        # Compute vector field
        X, Y, U, V = self.compute_vector_field(
            x_var, y_var, grid_size=grid_size, fixed_params=fixed_params
        )

        # Find equilibrium points (where both derivatives are close to zero)
        eq_points = []
        eq_types = []
        eq_eigenvalues = []
        eq_classifications = []  # Store more detailed classification

        for i in range(grid_size):
            for j in range(grid_size):
                if abs(U[i, j]) < 1e-2 and abs(V[i, j]) < 1e-2:
                    # Create full state vector
                    state = np.zeros(len(state_names))
                    state[vx_idx] = vx
                    state[r_idx] = X[i, j]
                    state[vy_idx] = Y[i, j]

                    # Verify with higher precision
                    derivatives = self.model.state_derivatives(0, state)
                    if (
                        abs(derivatives[r_idx]) < 1e-3
                        and abs(derivatives[vy_idx]) < 1e-3
                    ):
                        eq_points.append((X[i, j], Y[i, j]))

                        # Compute Jacobian for stability analysis
                        J = np.zeros((2, 2))
                        h = 1e-6

                        # Compute partial derivatives numerically
                        # dr/dr
                        state_p = state.copy()
                        state_p[r_idx] += h
                        der_p = self.model.state_derivatives(0, state_p)
                        J[0, 0] = (der_p[r_idx] - derivatives[r_idx]) / h

                        # dr/dvy
                        state_p = state.copy()
                        state_p[vy_idx] += h
                        der_p = self.model.state_derivatives(0, state_p)
                        J[0, 1] = (der_p[r_idx] - derivatives[r_idx]) / h

                        # dvy/dr
                        state_p = state.copy()
                        state_p[r_idx] += h
                        der_p = self.model.state_derivatives(0, state_p)
                        J[1, 0] = (der_p[vy_idx] - derivatives[vy_idx]) / h

                        # dvy/dvy
                        state_p = state.copy()
                        state_p[vy_idx] += h
                        der_p = self.model.state_derivatives(0, state_p)
                        J[1, 1] = (der_p[vy_idx] - derivatives[vy_idx]) / h

                        # Analyze eigenvalues for stability
                        eigvals = np.linalg.eigvals(J)
                        eq_eigenvalues.append(eigvals)

                        # Classify based on eigenvalues
                        real_parts = np.real(eigvals)
                        imag_parts = np.imag(eigvals)

                        # Detailed classification based on eigenvalue structure
                        if np.all(real_parts < 0):
                            if np.any(imag_parts != 0):
                                eq_types.append("Stable Focus")
                                eq_classifications.append("Stable Focus (Spiral Point)")
                            else:
                                eq_types.append("Stable Node")
                                eq_classifications.append("Stable Node")
                        elif np.all(real_parts > 0):
                            if np.any(imag_parts != 0):
                                eq_types.append("Unstable Focus")
                                eq_classifications.append(
                                    "Unstable Focus (Spiral Point)"
                                )
                            else:
                                eq_types.append("Unstable Node")
                                eq_classifications.append("Unstable Node")
                        else:
                            eq_types.append("Saddle Point")
                            eq_classifications.append("Saddle Point")

        # 1. Yaw rate vs lateral velocity with vector field
        ax1 = fig.add_subplot(gs[0, 0])

        # Normalize vectors for visualization
        norm = np.sqrt(U**2 + V**2)
        mask = norm > 1e-10
        U_norm = np.zeros_like(U)
        V_norm = np.zeros_like(V)
        U_norm[mask] = U[mask] / norm[mask]
        V_norm[mask] = V[mask] / norm[mask]

        # Draw streamlines
        strm = ax1.streamplot(
            X,
            Y,
            U_norm,
            V_norm,
            color=norm,
            cmap="viridis",
            linewidth=1.5,
            density=1.5,
            arrowsize=1.5,
        )
        plt.colorbar(strm.lines, ax=ax1, label="Vector magnitude")

        # Enhanced markers and colors for equilibrium points
        eq_markers = {
            "Stable Node": "o",
            "Stable Focus": "*",
            "Unstable Node": "s",
            "Unstable Focus": "p",
            "Saddle Point": "D",
        }

        eq_colors = {
            "Stable Node": "lime",
            "Stable Focus": "green",
            "Unstable Node": "red",
            "Unstable Focus": "orangered",
            "Saddle Point": "purple",
        }

        # Mark equilibrium points in first plot with enhanced visibility
        legend_elements = []
        for i, ((r_eq, vy_eq), eq_type) in enumerate(zip(eq_points, eq_types)):
            marker = eq_markers.get(eq_type, "o")
            color = eq_colors.get(eq_type, "black")

            # Plot equilibrium point with larger marker
            ax1.plot(
                r_eq,
                vy_eq,
                marker=marker,
                markersize=15,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1.5,
                label=f"Eq {i + 1}: {eq_type}",
                zorder=10,  # Ensure points are drawn on top
            )

            # Add equilibrium point number with enhanced visibility
            ax1.annotate(
                f"{i + 1}",
                (r_eq, vy_eq),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=14,
                weight="bold",
                color="white",
                bbox=dict(
                    facecolor=color,
                    alpha=0.9,
                    boxstyle="round,pad=0.3",
                    edgecolor="black",
                ),
                zorder=11,
            )

            # Add text with equilibrium point values
            beta_eq = np.rad2deg(
                np.arctan2(vy_eq, vx)
            )  # Convert to degrees for display
            value_text = f"r={r_eq:.3f}, vy={vy_eq:.3f}, β={beta_eq:.1f}°"

            # Position text with smart offset to avoid overlapping
            offset_x = 0.05 * r_range[1] if r_eq < 0 else -0.05 * r_range[1]
            offset_y = 0.03 * (vy_range[1] - vy_range[0])

            ax1.annotate(
                value_text,
                (r_eq, vy_eq),
                xytext=(r_eq + offset_x, vy_eq + offset_y),
                fontsize=10,
                bbox=dict(
                    facecolor="white",
                    alpha=0.8,
                    edgecolor=color,
                    boxstyle="round,pad=0.2",
                ),
                ha="center",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                zorder=9,
            )

            # Create legend elements for unique equilibrium types
            if eq_type not in [
                item.get_label().split(": ")[-1] for item in legend_elements
            ]:
                from matplotlib.lines import Line2D

                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker=marker,
                        color="w",
                        markerfacecolor=color,
                        markeredgecolor="black",
                        markersize=10,
                        label=eq_type,
                    )
                )

        ax1.set_xlabel("Yaw Rate r (rad/s)", fontsize=12)
        ax1.set_ylabel("Lateral Velocity vy (m/s)", fontsize=12)
        ax1.set_title(
            f"Phase Portrait: vy vs r (vx = {vx} m/s)", fontsize=14, fontweight="bold"
        )
        if legend_elements:
            ax1.legend(handles=legend_elements, loc="best", framealpha=0.9, fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # 2. Yaw acceleration contour plot
        ax2 = fig.add_subplot(gs[0, 1])

        # Create meshgrid for contour plot
        r_values = np.linspace(r_range[0], r_range[1], grid_size)
        vy_values = np.linspace(vy_range[0], vy_range[1], grid_size)
        R, VY = np.meshgrid(r_values, vy_values)

        # Initialize matrices for derivatives
        DR = np.zeros_like(R)
        DVY = np.zeros_like(VY)

        # Compute derivatives
        for i in range(grid_size):
            for j in range(grid_size):
                state = np.zeros(len(state_names))
                state[vx_idx] = vx
                state[vy_idx] = VY[i, j]
                state[r_idx] = R[i, j]

                derivatives = self.model.state_derivatives(0, state)
                DR[i, j] = derivatives[r_idx]
                DVY[i, j] = derivatives[vy_idx]

        # Plot yaw acceleration contours
        levels = np.linspace(-3.0, 3.0, 20)
        contour = ax2.contourf(R, VY, DR, levels=levels, cmap="coolwarm", extend="both")

        # Add contour lines for zero derivatives with clear labels
        dr_zero = ax2.contour(R, VY, DR, levels=[0], colors="k", linewidths=2)
        dvy_zero = ax2.contour(R, VY, DVY, levels=[0], colors="b", linewidths=2)

        plt.clabel(
            dr_zero,
            inline=True,
            fontsize=10,
            fmt="dr/dt=0",
            manual=[(r_range[0] * 0.8, vy_range[1] * 0.8)],
        )
        plt.clabel(
            dvy_zero,
            inline=True,
            fontsize=10,
            fmt="dvy/dt=0",
            manual=[(r_range[1] * 0.8, vy_range[0] * 0.8)],
        )

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax2)
        cbar.set_label("Yaw Acceleration (rad/s²)", fontsize=10)

        # Plot equilibrium points with enhanced visibility
        for i, ((r_eq, vy_eq), eq_type) in enumerate(zip(eq_points, eq_types)):
            marker = eq_markers.get(eq_type, "o")
            color = eq_colors.get(eq_type, "black")
            ax2.plot(
                r_eq,
                vy_eq,
                marker=marker,
                markersize=15,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=10,
            )
            ax2.annotate(
                f"{i + 1}",
                (r_eq, vy_eq),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=14,
                weight="bold",
                color="white",
                bbox=dict(
                    facecolor=color,
                    alpha=0.9,
                    boxstyle="round,pad=0.3",
                    edgecolor="black",
                ),
                zorder=11,
            )

            # Add values in a box with arrow
            beta_eq = np.rad2deg(np.arctan2(vy_eq, vx))
            value_text = f"r={r_eq:.3f}, vy={vy_eq:.3f}\nβ={beta_eq:.1f}°"

            # Position box with values in less crowded areas
            offset_x = -0.2 * r_range[1] if r_eq > 0 else 0.2 * r_range[1]
            offset_y = -0.1 * vy_range[1] if vy_eq > 0 else 0.1 * vy_range[1]

            ax2.annotate(
                value_text,
                (r_eq, vy_eq),
                xytext=(r_eq + offset_x, vy_eq + offset_y),
                fontsize=9,
                bbox=dict(
                    facecolor="white",
                    alpha=0.9,
                    edgecolor=color,
                    boxstyle="round,pad=0.2",
                ),
                ha="center",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                zorder=9,
            )

        ax2.set_xlabel("Yaw Rate r (rad/s)", fontsize=12)
        ax2.set_ylabel("Lateral Velocity vy (m/s)", fontsize=12)
        ax2.set_title(
            f"Yaw Acceleration & Zero-Derivative Lines (vx = {vx} m/s)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.grid(True, linestyle="--", alpha=0.6)

        # 3. Lateral acceleration contour plot
        ax3 = fig.add_subplot(gs[1, 0])

        # Plot lateral acceleration contours
        levels = np.linspace(-3.0, 3.0, 20)
        contour = ax3.contourf(
            R, VY, DVY, levels=levels, cmap="coolwarm", extend="both"
        )

        # Add the same contour lines with clearer labels
        dr_zero = ax3.contour(R, VY, DR, levels=[0], colors="k", linewidths=2)
        dvy_zero = ax3.contour(R, VY, DVY, levels=[0], colors="b", linewidths=2)

        plt.clabel(
            dr_zero,
            inline=True,
            fontsize=10,
            fmt="dr/dt=0",
            manual=[(r_range[0] * 0.8, vy_range[1] * 0.8)],
        )
        plt.clabel(
            dvy_zero,
            inline=True,
            fontsize=10,
            fmt="dvy/dt=0",
            manual=[(r_range[1] * 0.8, vy_range[0] * 0.8)],
        )

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax3)
        cbar.set_label("Lateral Acceleration (m/s²)", fontsize=10)

        # Plot equilibrium points with enhanced visibility
        for i, ((r_eq, vy_eq), eq_type) in enumerate(zip(eq_points, eq_types)):
            marker = eq_markers.get(eq_type, "o")
            color = eq_colors.get(eq_type, "black")
            ax3.plot(
                r_eq,
                vy_eq,
                marker=marker,
                markersize=15,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=10,
            )
            ax3.annotate(
                f"{i + 1}",
                (r_eq, vy_eq),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=14,
                weight="bold",
                color="white",
                bbox=dict(
                    facecolor=color,
                    alpha=0.9,
                    boxstyle="round,pad=0.3",
                    edgecolor="black",
                ),
                zorder=11,
            )

            # Add eigenvalue information for each point
            eigvals = eq_eigenvalues[i]
            eig_real = np.real(eigvals)
            eig_imag = np.imag(eigvals)

            eig_text = f"λ₁={eig_real[0]:.3f}"
            if abs(eig_imag[0]) > 1e-10:
                eig_text += f"±{abs(eig_imag[0]):.3f}j"

            eig_text += f"\nλ₂={eig_real[1]:.3f}"
            if abs(eig_imag[1]) > 1e-10 and abs(eig_imag[0]) < 1e-10:
                eig_text += f"±{abs(eig_imag[1]):.3f}j"

            # Smart positioning to avoid overlaps
            offset_x = 0.25 * r_range[1] if i % 2 == 0 else -0.25 * r_range[1]
            offset_y = 0.15 * vy_range[1] if i % 3 == 0 else -0.15 * vy_range[1]

            ax3.annotate(
                eig_text,
                (r_eq, vy_eq),
                xytext=(r_eq + offset_x, vy_eq + offset_y),
                fontsize=9,
                bbox=dict(
                    facecolor="white",
                    alpha=0.9,
                    edgecolor=color,
                    boxstyle="round,pad=0.2",
                ),
                ha="center",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                zorder=9,
            )

        ax3.set_xlabel("Yaw Rate r (rad/s)", fontsize=12)
        ax3.set_ylabel("Lateral Velocity vy (m/s)", fontsize=12)
        ax3.set_title(
            f"Lateral Acceleration & Eigenvalues (vx = {vx} m/s)",
            fontsize=14,
            fontweight="bold",
        )
        ax3.grid(True, linestyle="--", alpha=0.6)

        # 4. Stability analysis with trajectory simulation
        ax4 = fig.add_subplot(gs[1, 1])

        # Display vector field
        quiv = ax4.quiver(
            X[::2, ::2],
            Y[::2, ::2],
            U_norm[::2, ::2],
            V_norm[::2, ::2],
            norm[::2, ::2],
            cmap="viridis",
            scale=25,
            width=0.002,
        )
        plt.colorbar(quiv, ax=ax4, label="Vector magnitude")

        # Plot zero-derivative lines with clear labels
        ax4.contour(R, VY, DR, levels=[0], colors="k", linewidths=2)
        ax4.contour(R, VY, DVY, levels=[0], colors="b", linewidths=2)

        # Add text labels for the curves
        ax4.text(
            r_range[0] * 0.7,
            vy_range[1] * 0.7,
            "dr/dt=0",
            color="k",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"),
        )
        ax4.text(
            r_range[1] * 0.7,
            vy_range[0] * 0.7,
            "dvy/dt=0",
            color="blue",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="blue"),
        )

        # Plot equilibrium points with clear type labels
        for i, ((r_eq, vy_eq), eq_type) in enumerate(zip(eq_points, eq_types)):
            marker = eq_markers.get(eq_type, "o")
            color = eq_colors.get(eq_type, "black")
            ax4.plot(
                r_eq,
                vy_eq,
                marker=marker,
                markersize=15,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=10,
            )

            # Number label
            ax4.annotate(
                f"{i + 1}",
                (r_eq, vy_eq),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=14,
                weight="bold",
                color="white",
                bbox=dict(
                    facecolor=color,
                    alpha=0.9,
                    boxstyle="round,pad=0.3",
                    edgecolor="black",
                ),
                zorder=11,
            )

            # Add clear type label with values
            beta_eq = np.rad2deg(np.arctan2(vy_eq, vx))
            type_text = f"{eq_type}\nr={r_eq:.3f}, vy={vy_eq:.3f}, β={beta_eq:.1f}°"

            # Strategic positioning based on equilibrium point number
            theta = 2 * np.pi * (i / len(eq_points))
            offset_dist = 0.2 * min(
                abs(r_range[1] - r_range[0]), abs(vy_range[1] - vy_range[0])
            )
            offset_x = offset_dist * np.cos(theta)
            offset_y = offset_dist * np.sin(theta)

            ax4.annotate(
                type_text,
                (r_eq, vy_eq),
                xytext=(r_eq + offset_x, vy_eq + offset_y),
                fontsize=9,
                bbox=dict(
                    facecolor="white",
                    alpha=0.9,
                    edgecolor=color,
                    boxstyle="round,pad=0.3",
                ),
                ha="center",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                zorder=9,
            )

        # Simulate trajectories from various initial conditions
        if eq_points:
            # Generate initial conditions around equilibrium points and other regions
            init_conditions = []
            for r_eq, vy_eq in eq_points:
                # Points around equilibrium
                for offset_r in [-0.3, 0.3]:
                    for offset_vy in [-2.0, 2.0]:
                        init_conditions.append([r_eq + offset_r, vy_eq + offset_vy])

            # Add some additional points
            init_conditions.extend(
                [
                    [r_range[0] + 0.1, 0],
                    [r_range[1] - 0.1, 0],
                    [0, vy_range[0] + 1.0],
                    [0, vy_range[1] - 1.0],
                ]
            )

            # Simulate trajectories
            trajectories = self.compute_trajectories(
                x_var, y_var, init_conditions, (0, 5.0), fixed_params
            )

            # Plot trajectories with different colors
            colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
            self.plot_trajectories(ax4, trajectories, x_var, y_var, colors)

        ax4.set_xlabel("Yaw Rate r (rad/s)", fontsize=12)
        ax4.set_ylabel("Lateral Velocity vy (m/s)", fontsize=12)
        ax4.set_title(
            f"Stability Analysis with Trajectories (vx = {vx} m/s)",
            fontsize=14,
            fontweight="bold",
        )
        ax4.grid(True, linestyle="--", alpha=0.6)

        # Create a comprehensive table for equilibrium points
        if eq_points:
            # Add a detailed table of equilibrium points at the bottom
            table_data = []
            table_columns = [
                "#",
                "r (rad/s)",
                "vy (m/s)",
                "β (deg)",
                "Type",
                "λ₁",
                "λ₂",
            ]

            for i, ((r_eq, vy_eq), eq_type, eigvals) in enumerate(
                zip(eq_points, eq_classifications, eq_eigenvalues)
            ):
                beta_eq = np.arctan2(vy_eq, vx)
                beta_deg = np.degrees(beta_eq)

                # Format eigenvalues for display
                eig1 = complex(eigvals[0])
                eig2 = complex(eigvals[1])

                # Round for display and handle complex numbers
                if abs(eig1.imag) < 1e-10:
                    eig1_str = f"{eig1.real:.4f}"
                else:
                    eig1_str = f"{eig1.real:.4f} + {eig1.imag:.4f}j"

                if abs(eig2.imag) < 1e-10:
                    eig2_str = f"{eig2.real:.4f}"
                else:
                    eig2_str = f"{eig2.real:.4f} + {eig2.imag:.4f}j"

                table_data.append(
                    [
                        i + 1,
                        f"{r_eq:.4f}",
                        f"{vy_eq:.4f}",
                        f"{beta_deg:.2f}",
                        eq_type,
                        eig1_str,
                        eig2_str,
                    ]
                )

            # Create a figure-wide table with equilibrium point details
            ax_table = fig.add_axes([0.1, 0.01, 0.8, 0.1])
            ax_table.axis("off")
            table = ax_table.table(
                cellText=table_data,
                colLabels=table_columns,
                loc="center",
                cellLoc="center",
                colWidths=[0.05, 0.12, 0.12, 0.12, 0.25, 0.17, 0.17],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

            # Style the table with more distinctive colors
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_facecolor("lightgray")
                    cell.set_text_props(weight="bold")
                elif j == 0:  # Index column
                    cell.set_facecolor("whitesmoke")
                elif j == 4:  # Type column
                    cell_text = cell.get_text().get_text()
                    if "Stable Node" in cell_text:
                        cell.set_facecolor("lime")
                    elif "Stable Focus" in cell_text:
                        cell.set_facecolor("lightgreen")
                    elif "Saddle" in cell_text:
                        cell.set_facecolor("plum")
                    elif "Unstable Node" in cell_text:
                        cell.set_facecolor("salmon")
                    elif "Unstable Focus" in cell_text:
                        cell.set_facecolor("orangered")

            # Add text annotation for table
            fig.text(
                0.5,
                0.12,
                "Equilibrium Points Analysis",
                horizontalalignment="center",
                fontsize=14,
                fontweight="bold",
            )

        # Add main title to the entire figure
        fig.suptitle(
            f"Vehicle Stability Analysis at vx = {vx} m/s",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Adjust layout to make room for the table and better spacing
        plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.3, top=0.92)

        return fig, (ax1, ax2, ax3, ax4), eq_points, eq_types, eq_eigenvalues


# %% Main function to run the analysis
if __name__ == "__main__":
    ### Basic Example Usage ###
    # Create vehicle model
    model = DualTrackBicycleModel()

    # Create analyzer
    analyzer = PhasePortraitAnalyzer(model)

    # Perform comprehensive analysis at a specific speed
    fig = analyzer.comprehensive_analysis(
        vx=30.0, r_range=(-5.0, 5.0), vy_range=(-20.0, 20.0), grid_size=1000
    )
    plt.show()

"""     fig = analyzer.yaw_acceleration_analysis(
        (-5.0, 5.0), (-10.0, 10.0), vx=30.0, grid_size=100
    )
    plt.show() """

"""     ### Custom Example Usage ###
    # Create custom model with different parameters
    custom_params = {
        "m": 2000.0,
        "Iz": 4000.0,
        "lf": 1.5,
        "lr": 1.3,
        "Cf": 100000.0,
        "Cr": 120000.0,
        "mu": 0.8,
    }
    custom_model = DualTrackBicycleModel(params=custom_params)
    custom_analyzer = PhasePortraitAnalyzer(custom_model)

    # Analyze custom model
    fig = custom_analyzer.comprehensive_analysis(vx=15.0)
    plt.show() """

"""
Brusselator Problem.
Internal and External Differentiation.
Betts, "Practical Methods for Optimal Control and Estimation Using Nonlinear Programming", 3rd Edition, 2020.
"""

# %% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation


# %% Brusselator Problem
# Define the Brusselator ODE system
def brusselator(t, y, a, b):
    """
    Brusselator ODE system:
    dx/dt = a + x^2*y - (b+1)*x
    dy/dt = b*x - x^2*y

    Parameters:
        t: time
        y: state vector [x, y]
        a, b: model parameters
    """
    x, y = y
    dx_dt = a + x**2 * y - (b + 1) * x
    dy_dt = b * x - x**2 * y
    return [dx_dt, dy_dt]


# %% Integrate ODE using Eighth-Order Dormand-Prince Runge-Kutta Method (DOP853)
# Set parameters
a = 1.0
b = 3.0
t_span = (0, 20)
y0 = [1.5, 2.91]  # Initial conditions

# Solve the ODE using DOP853 (8th-order Dormand-Prince method)
solution = solve_ivp(
    fun=lambda t, y: brusselator(t, y, a, b),
    t_span=t_span,
    y0=y0,
    method="DOP853",  # 8th-order Dormand-Prince method
    rtol=1e-8,  # Relative tolerance
    atol=1e-8,  # Absolute tolerance
    dense_output=True,  # Enable dense output for smooth plotting
)

# Create time points for smooth plotting
t_dense = np.linspace(t_span[0], t_span[1], 1000)
y_dense = solution.sol(t_dense)
x_dense, y_dense = y_dense[0], y_dense[1]

# Create phase plot
plt.figure(figsize=(12, 10))

# Plot time evolution
plt.subplot(2, 2, 1)
plt.plot(t_dense, x_dense, "b-", label="x(t)")
plt.plot(t_dense, y_dense, "r-", label="y(t)")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Time Evolution of Brusselator")
plt.legend()
plt.grid(True)

# Plot phase portrait
plt.subplot(2, 2, 2)
plt.plot(x_dense, y_dense, "g-")
plt.plot(x_dense[0], y_dense[0], "ro", label="Start")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Phase Portrait")
plt.grid(True)
plt.legend()

# Plot x component
plt.subplot(2, 2, 3)
plt.plot(t_dense, x_dense, "b-")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.title("x Component")
plt.grid(True)

# Plot y component
plt.subplot(2, 2, 4)
plt.plot(t_dense, y_dense, "r-")
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.title("y Component")
plt.grid(True)

plt.tight_layout()
plt.show()


# %% Create an animation of the phase portrait
def create_animation():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(min(x_dense) - 0.1, max(x_dense) + 0.1)
    ax.set_ylim(min(y_dense) - 0.1, max(y_dense) + 0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Brusselator Phase Portrait Animation")
    ax.grid(True)

    (line,) = ax.plot([], [], "g-", lw=2)
    (point,) = ax.plot([], [], "ro", ms=8)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        line.set_data([], [])
        point.set_data([], [])
        time_text.set_text("")
        return line, point, time_text

    def animate(i):
        # Use a growing segment of the trajectory
        idx = i * 5  # Adjust speed of animation
        if idx >= len(x_dense):
            idx = len(x_dense) - 1

        # Important fix: Use arrays for both x and y values, not single values
        line.set_data(x_dense[: idx + 1], y_dense[: idx + 1])

        # Fix: Make sure we're passing sequences (arrays), not single values
        point.set_data([x_dense[idx]], [y_dense[idx]])  # Use lists to ensure sequences

        time_text.set_text(f"Time = {t_dense[idx]:.2f}")
        return line, point, time_text

    # Create animation
    ani = FuncAnimation(
        fig, animate, frames=len(t_dense) // 5, init_func=init, blit=True, interval=20
    )

    return ani, fig


# %% Function to analyze stability and behavior for different parameters
def analyze_brusselator_stability(a_range, b_range):
    """
    Analyze the stability of the Brusselator system for different parameter values.
    The system has a fixed point at (x*, y*) = (a, b/a).
    It undergoes a Hopf bifurcation when b > 1 + a^2.
    """
    a_values = np.linspace(a_range[0], a_range[1], 100)
    b_values = np.linspace(b_range[0], b_range[1], 100)

    A, B = np.meshgrid(a_values, b_values)
    stability = B > 1 + A**2

    plt.figure(figsize=(10, 8))
    plt.contourf(A, B, stability, levels=[0, 0.5, 1], colors=["lightblue", "salmon"])
    plt.colorbar(ticks=[0.25, 0.75], label="System Behavior")
    plt.contour(A, B, B - (1 + A**2), levels=[0], colors="k")
    plt.xlabel("Parameter a")
    plt.ylabel("Parameter b")
    plt.title("Brusselator Stability Diagram")

    # Add labels for the regions
    plt.text(0.5, 0.5, "Stable\nFixed Point", ha="center", fontsize=12)
    plt.text(0.5, 2.5, "Limit Cycle\n(Oscillations)", ha="center", fontsize=12)

    # Mark the current parameters
    plt.plot(a, b, "ko", markersize=8)
    plt.text(a + 0.1, b + 0.1, f"Current (a={a}, b={b})", fontsize=10)

    plt.grid(True)
    plt.show()


# %% Function to explore different initial conditions
def explore_initial_conditions(X=None, tol=1e-8):
    """
    Integrate the system with various initial conditions to visualize
    the basin of attraction for the limit cycle.
    """

    nx = len(X)
    y1_fixed = 1.5
    nt = 1000
    ySol = np.zeros((2, nx))  # initialize array to store solutions
    plt.figure(figsize=(10, 8))

    for i, y2 in enumerate(X):
        y0_test = [y1_fixed, y2]
        sol = solve_ivp(
            fun=lambda t, y: brusselator(t, y, a, b),
            t_span=t_span,
            y0=y0_test,
            method="DOP853",
            rtol=tol,
            atol=tol,
            dense_output=True,
        )

        t_test = np.linspace(t_span[0], t_span[1], nt)
        y_test = sol.sol(t_test)
        ySol[:, i] = y_test[:, -1]  # store the last point of the solution

        plt.plot(y_test[0], y_test[1], "-", linewidth=1, alpha=0.7)
        plt.plot(y_test[0][0], y_test[1][0], "o", markersize=4)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Multiple Initial Conditions for Brusselator")
    plt.grid(True)
    plt.show()

    return ySol


# %% Usage examples:
if __name__ == "__main__":
    # First generate the standard plots
    # plt.show() is already included in the plotting code

    # To run stability analysis
    # analyze_brusselator_stability((0, 2), (0, 5))

    # To run the animation
    # ani, fig = create_animation()
    # plt.show()

    # To explore different initial conditions
    X = np.arange(2.90, 3.1, 0.01)
    tol = 1e-4
    yNom = explore_initial_conditions(X, tol)  # Solution at Nominal Points

    # Perturbations
    yPert = np.zeros((2, len(X)))
    G_tF = np.zeros(
        (2, len(X))
    )  # Computing derivative of ODE solution w.r.t variables x
    for k in range(1, len(X)):
        delta = 10 ** (-3) * X[k]
        xBar = X[k] + delta
        yPert[:, k] = explore_initial_conditions(np.array([xBar]), tol).squeeze()
        G_tF[:, k] = (yPert[:, k] - yNom[:, k]) / delta

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(X, G_tF[0], linewidth=1, label="dx/dt")
    plt.plot(X, G_tF[1], linewidth=1, label="dy/dt")
    plt.xlabel("Y2 Initial Condition")
    plt.ylabel("Gradient")
    plt.title("External Derivatives (1e-4)")
    plt.legend()
    plt.grid(True)
    plt.show()


# %%

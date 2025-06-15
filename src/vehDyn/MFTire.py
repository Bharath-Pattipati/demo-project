# %% Import Libraries
import numpy as np
from pysr import PySRRegressor
import matplotlib.pyplot as plt


# %% MF Tire Model
class MFSwiftTire:
    def __init__(self, params):
        """
        Initialize the MF-Swift tire model with parameters.
        :param params: Dictionary containing Magic Formula coefficients.
        """
        self.params = params

    def magic_formula(self, slip, B, C, D, E):
        """
        Evaluate the Magic Formula equation.
        :param slip: Slip ratio or slip angle (in radians).
        :param B: Stiffness factor.
        :param C: Shape factor.
        :param D: Peak factor.
        :param E: Curvature factor.
        :return: Force (longitudinal or lateral).
        """
        return D * np.sin(
            C * np.arctan(B * slip - E * (B * slip - np.arctan(B * slip)))
        )

    def calc_longitudinal_force(self, kappa):
        """
        Calculate longitudinal force Fx using Magic Formula.
        :param kappa: Longitudinal slip ratio.
        :return: Longitudinal force Fx.
        """
        Bx = self.params["Bx"]
        Cx = self.params["Cx"]
        Dx = self.params["Dx"]
        Ex = self.params["Ex"]

        return self.magic_formula(kappa, Bx, Cx, Dx, Ex)

    def calc_lateral_force(self, alpha):
        """
        Calculate lateral force Fy using Magic Formula.
        :param alpha: Slip angle (in radians).
        :return: Lateral force Fy.
        """
        By = self.params["By"]
        Cy = self.params["Cy"]
        Dy = self.params["Dy"]
        Ey = self.params["Ey"]

        return self.magic_formula(alpha, By, Cy, Dy, Ey)

    def calc_aligning_moment(self, alpha):
        """
        Calculate aligning moment Mz using Magic Formula.
        :param alpha: Slip angle (in radians).
        :return: Aligning moment Mz.
        """
        Bz = self.params["Bz"]
        Cz = self.params["Cz"]
        Dz = self.params["Dz"]
        Ez = self.params["Ez"]

        return self.magic_formula(alpha, Bz, Cz, Dz, Ez)


# %% Example usage
""" if __name__ == "__main__":
    # Example parameter set for a tire
    tire_params = {
        "Bx": 10.0,
        "Cx": 1.9,
        "Dx": 1.0,
        "Ex": 0.97,
        "By": 8.0,
        "Cy": 1.8,
        "Dy": 1.2,
        "Ey": 1.0,
        "Bz": 5.0,
        "Cz": 2.0,
        "Dz": 0.5,
        "Ez": 1.1,
    }

    tire_model = MFSwiftTire(tire_params)

    # Longitudinal force vs. slip ratio
    kappa_values = np.linspace(-0.2, 0.2, 100)
    Fx_values = [tire_model.calc_longitudinal_force(kappa) for kappa in kappa_values]

    # Lateral force vs. slip angle
    alpha_values = np.linspace(-np.pi / 6, np.pi / 6, 100)  # +/-30 degrees
    Fy_values = [tire_model.calc_lateral_force(alpha) for alpha in alpha_values]

    # Plot results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(kappa_values, Fx_values)
    plt.title("Longitudinal Force (Fx)")
    plt.xlabel("Slip Ratio (kappa)")
    plt.ylabel("Force Fx (N)")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(np.degrees(alpha_values), Fy_values)
    plt.title("Lateral Force (Fy)")
    plt.xlabel("Slip Angle (degrees)")
    plt.ylabel("Force Fy (N)")
    plt.grid()

    plt.tight_layout()
    plt.show() """

# %% PYSR Model Identification
# Example parameter set for a tire
tire_params = {
    "Bx": 10.0,
    "Cx": 1.9,
    "Dx": 1.0,
    "Ex": 0.97,
    "By": 8.0,
    "Cy": 1.8,
    "Dy": 1.2,
    "Ey": 1.0,
    "Bz": 5.0,
    "Cz": 2.0,
    "Dz": 0.5,
    "Ez": 1.1,
}

tire_model = MFSwiftTire(tire_params)

N = 200  # number of data points

# Longitudinal force vs. slip ratio
kappa_values = np.linspace(-0.2, 0.2, N)
Fx_values = [tire_model.calc_longitudinal_force(kappa) for kappa in kappa_values]

# Lateral force vs. slip angle
alpha_values = np.linspace(-np.pi / 6, np.pi / 6, N)  # +/-30 degrees
Fy_values = [tire_model.calc_lateral_force(alpha) for alpha in alpha_values]

# Create training data: X(n_samples,n_features), y(n_samples,n_targets)
X = np.column_stack(
    (kappa_values, alpha_values)
)  # kappa_values.reshape(-1, 1), alpha_values.reshape(-1, 1)
y = np.column_stack((Fx_values, Fy_values))  # Fx_values, Fy_values

# PySRRegressor Parameters
pysr_params = dict(
    populations=50,
    model_selection="best",
    maxsize=30,  # max complexity of an equation
    maxdepth=None,  # max depth of an equation
    optimizer_algorithm="BFGS",  # NelderMead or BFGS
    nested_constraints={
        "atan": {"atan": 1},
        "sin": {"sin": 0},
    },  # Specifies how many times a combination of operators can be nested
    parsimony=10,  # Multiplicative factor to punish complexity.
)

# Learn equations
tirModel = PySRRegressor(
    niterations=50,
    binary_operators=["+", "-", "*"],
    unary_operators=["sin", "exp", "atan", "log"],
    **pysr_params,
)

tirModel.fit(X, y)
print("Best Tire Model Equations for Fx and Fy: \n", tirModel.sympy())

# filter equations based on loss and then select the best score from that list
""" threshold = 2 * min([eq.loss.min() for eq in tirModel.equations_])
scores = [eq.score for eq in tirModel.equations_ if eq.loss.min() < threshold]
best_idx = [score.idxmax() for score in scores]

if best_idx is not None:
    print("Best equation: ", tirModel.sympy(best_idx))
else:
    print("No equations found with loss less than the threshold.")

y_predict = tirModel.predict(X, index=best_idx) """

y_predict = tirModel.predict(X)
print("MSE: ", np.power(y_predict - y, 2).mean())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(X[:, 0], y[:, 0], label="Truth")
plt.plot(X[:, 0], y_predict[:, 0], "r", label="Predicted")
plt.title("Longitudinal Force (Fx)")
plt.xlabel(r"$Slip Ratio (\kappa)$")
plt.ylabel(r"$F_{X} (N)$")
plt.legend()
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(X[:, 1], y[:, 1], label="Truth")
plt.plot(X[:, 1], y_predict[:, 1], "r", label="Predicted")
plt.title("Lateral Force (Fy)")
plt.xlabel(r"$Slip Angle (\alpha)$")
plt.ylabel(r"$F_{Y} (N)$")
plt.legend()
plt.grid()
plt.show()

# %%

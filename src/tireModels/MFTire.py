import numpy as np


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


# Example usage
if __name__ == "__main__":
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
    plt.show()

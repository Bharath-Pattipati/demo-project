# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt

# import PAC2002
from PAC2002.PAC2002 import PAC2002
from PAC2002.PAC2002 import TireState


# %% Dugoff Tire Model
# class DugoffTire:
# def __init__(self, Ca=100000, Ck=1000000, mu=0.8, Fz=4500):
"""
        Initialize the Dugoff tire model.
        Parameters:
        - Ca: Cornering stiffness (Nm/rad)
        - Ck: Longitudinal stiffness
        - mu: Friction coefficient
        - Fz: Vertical load (N)
"""
# self.Ca = Ca
# self.Ck = Ck
# self.mu = mu
# self.Fz = Fz

# def calc_forces(self, kappa, alpha):
"""
        Calculate longitudinal (Fx) and lateral (Fy) forces.
        Parameters:
        - kappa: Longitudinal slip ratio
        - alpha: Slip angle (in radians)

        Returns:
        - Fx: Longitudinal force (N)
        - Fy: Lateral force (N)
"""


"""         sig_x = kappa / (1 + kappa)
        sig_y = np.tan(alpha) / (1 + kappa)

        Fx_lin = self.Ck * sig_x
        Fy_lin = self.Ca * sig_y

        combined_force = np.sqrt(Fx_lin**2 + Fy_lin**2)

        if combined_force <= self.mu * self.Fz:
            Fx, Fy = Fx_lin, Fy_lin
        else:
            lam = (self.mu * self.Fz) / combined_force
            Fx = Fx_lin * lam
            Fy = Fy_lin * lam

        return Fx, Fy """


# %% Example usage and visualization
if __name__ == "__main__":
    # iterate over different vertical loads and plot them in the same figure
    vertical_loads = [3000, 4000, 5000, 6000, 7000]

    # Define a list of colors
    colors = ["red", "green", "blue", "orange", "purple"]

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(2, figsize=(8, 6))

    # PAC2002 Tire Model
    pacTireMdl = PAC2002()
    pacTireMdl.createmodel()
    state = TireState()
    state["FZ"] = 1500
    state["IA"] = 0.0
    state["SR"] = 0.0
    state["SA"] = 0.0
    state["FY"] = 0.0
    state["V"] = 10.0
    state["P"] = 260000

    for i, Fz in enumerate(vertical_loads):
        # tire_model = DugoffTire(Fz=Fz)

        # Slip angle range for lateral force calculation
        alpha_range = (
            np.linspace(-15, 15, 100) * np.pi / 180
        )  # 0 to 15 degrees in radians
        # Fy_values = [tire_model.calc_forces(0, alpha)[1] for alpha in alpha_range]

        # Slip ratio range for longitudinal force calculation
        kappa_range = np.linspace(-0.2, 0.2, 100)  # 0 to 20%
        # Fx_values = [tire_model.calc_forces(kappa, 0)[0] for kappa in kappa_range]

        # PAC2002 Tire Model
        Fy_pac = []
        for sa in alpha_range:
            state["SA"] = sa
            state["FZ"] = Fz
            pacState = pacTireMdl.solve(state)
            Fy_pac = np.append(Fy_pac, pacState["FY"])

        axs[0].plot(
            np.degrees(alpha_range),
            Fy_pac,
            color=colors[i],
            label=f"Fz = {vertical_loads[i]}",
        )

        # PAC2002 Tire Model
        Fx_pac = []
        for sr in kappa_range:
            state["SR"] = sr
            state["FZ"] = Fz
            pacState = pacTireMdl.solve(state)
            Fx_pac = np.append(Fx_pac, pacState["FX"])

        axs[1].plot(
            kappa_range,
            Fx_pac,
            color=colors[i],
            label=f"Fz = {vertical_loads[i]}",
        )

    axs[0].set_xlabel("Slip Angle (degrees)")
    axs[0].set_ylabel("Lateral Force (N)")
    axs[0].grid()
    axs[0].legend()
    axs[1].set_xlabel("Slip Ratio")
    axs[1].set_ylabel("Longitudinal Force (N)")
    axs[1].grid()
    axs[1].legend()
    plt.show()

# %%

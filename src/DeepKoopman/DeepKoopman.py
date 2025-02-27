"""
DeepKoopman, Bethany Lusch, Brunton, Kutz, Github: https://github.com/BethanyL/DeepKoopman
Data: https://anl.app.box.com/s/9s29juzu892dfkhgxa1n1q4mj63nxabn?sortColumn=name&sortDirection=ASC
"""

# %% import libraries
import os
import helperfcns as hf

import tensorflow as tf
# from tensorflow import keras

import numpy as np
import KoopmanArch as net

# %% setup parameters for DeepKoopman
params = {}

# settings related to dataset
params["data_name"] = "DiscreteSpectrumExample"
params["data_train_len"] = 1
params["len_time"] = 51
n = 2  # dimension of system (and input layer)
num_initial_conditions = 5000  # per training file
params["delta_t"] = 0.02

# settings related to saving results
params["folder_name"] = "exp1_best"

# settings related to network architecture
params["num_real"] = 2
params["num_complex_pairs"] = 0
params["num_evals"] = 2
k = params["num_evals"]  # dimension of y-coordinates
w = 30
params["widths"] = [2, w, w, k, k, w, w, 2]
wo = 10
params["hidden_widths_omega"] = [wo, wo, wo]

# settings related to loss function
params["num_shifts"] = 30
params["num_shifts_middle"] = params["len_time"] - 1
max_shifts = max(params["num_shifts"], params["num_shifts_middle"])
num_examples = num_initial_conditions * (params["len_time"] - max_shifts)
params["recon_lam"] = 0.1
params["Linf_lam"] = 10 ** (-7)
params["L1_lam"] = 0.0
params["L2_lam"] = 10 ** (-15)
params["auto_first"] = 0

# settings related to the training
params["num_passes_per_file"] = 15 * 6 * 10
params["num_steps_per_batch"] = 2
params["learning_rate"] = 10 ** (-3)
params["batch_size"] = 256
steps_to_see_all = num_examples / params["batch_size"]
params["num_steps_per_file_pass"] = (int(steps_to_see_all) + 1) * params[
    "num_steps_per_batch"
]

# settings related to the timing
params["max_time"] = 4 * 60 * 60  # 4 hours
params["min_5min"] = 0.5
params["min_20min"] = 0.0004
params["min_40min"] = 0.00008
params["min_1hr"] = 0.00003
params["min_2hr"] = 0.00001
params["min_3hr"] = 0.000006
params["min_halfway"] = 0.000006


# %% Create Koopman Network
def KoopmanNet(data, params):
    # setup network
    x, y, g_list, weights, biases = net.create_koopman_network(params)


# %% Main experiments function
def main(params):
    hf.set_defaults(params)

    if not os.path.exists(params["folder_name"]):
        os.makedirs(params["folder_name"])

    tf.random.set_seed(params["seed"])
    np.random.seed(params["seed"])

    # data is num_steps x num_examples x n but load flattened version (matrix instead of tensor)
    data_val = np.loadtxt(
        ("../demo-project/data/processed/%s_val_x.csv" % (params["data_name"])),
        delimiter=",",
        dtype=np.float64,
    )

    KoopmanNet(data_val, params)


# %% Execute Main Function
if __name__ == "__main__":
    main(params)

# %%

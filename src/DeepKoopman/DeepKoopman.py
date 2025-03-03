"""
DeepKoopman, Bethany Lusch, Brunton, Kutz, Github: https://github.com/BethanyL/DeepKoopman
Data: https://anl.app.box.com/s/9s29juzu892dfkhgxa1n1q4mj63nxabn?sortColumn=name&sortDirection=ASC
"""

# %% import libraries
import os
import time
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


# %% Custom Loss Functions
def define_loss(x, y, g_list, weights, biases, params):
    """Define the (unregularized) loss functions for the training.

    Arguments:
        x -- placeholder for input
        y -- list of outputs of network for each shift (each prediction step)
        g_list -- list of output of encoder for each shift (encoding each step in x)
        weights -- dictionary of weights for all networks
        biases -- dictionary of biases for all networks
        params -- dictionary of parameters for experiment

    Returns:
        loss1 -- autoencoder loss function
        loss2 -- dynamics/prediction loss function
        loss3 -- linearity loss function
        loss_Linf -- inf norm on autoencoder loss and one-step prediction loss
        loss -- sum of above four losses

    Side effects:
        None
    """
    # Minimize the mean squared errors.
    # subtraction and squaring element-wise, then average over both dimensions
    # n columns
    # average of each row (across columns), then average the rows
    denominator_nonzero = 10 ** (-5)

    # autoencoder reconstruction loss
    if params["relative_loss"]:
        loss1_denominator = (
            tf.reduce_mean(tf.reduce_mean(tf.square(tf.squeeze(x[0, :, :])), 1))
            + denominator_nonzero
        )
    else:
        loss1_denominator = tf.to_double(1.0)

    mean_squared_error = tf.reduce_mean(
        tf.reduce_mean(tf.square(y[0] - tf.squeeze(x[0, :, :])), 1)
    )
    loss1 = params["recon_lam"] * tf.truediv(mean_squared_error, loss1_denominator)

    # prediction loss
    loss2 = tf.zeros(
        [
            1,
        ],
        dtype=tf.float64,
    )
    if params["num_shifts"] > 0:
        for j in np.arange(params["num_shifts"]):
            # xk+1, xk+2, xk+3
            shift = params["shifts"][j]
            if params["relative_loss"]:
                loss2_denominator = (
                    tf.reduce_mean(
                        tf.reduce_mean(tf.square(tf.squeeze(x[shift, :, :])), 1)
                    )
                    + denominator_nonzero
                )
            else:
                loss2_denominator = tf.to_double(1.0)
            loss2 = loss2 + params["recon_lam"] * tf.truediv(
                tf.reduce_mean(
                    tf.reduce_mean(tf.square(y[j + 1] - tf.squeeze(x[shift, :, :])), 1)
                ),
                loss2_denominator,
            )
        loss2 = loss2 / params["num_shifts"]

    # Linearity loss
    loss3 = tf.zeros(
        [
            1,
        ],
        dtype=tf.float64,
    )
    count_shifts_middle = 0
    if params["num_shifts_middle"] > 0:
        # generalization of: next_step = tf.matmul(g_list[0], L_pow)
        KoopmanOmega = net.Omega(params, g_list[0])
        omegas = KoopmanOmega.omega_net_apply(params, g_list[0], weights, biases)
        next_step = net.varying_multiply(
            g_list[0],
            omegas,
            params["delta_t"],
            params["num_real"],
            params["num_complex_pairs"],
        )
        # multiply g_list[0] by L (j+1) times
        for j in np.arange(max(params["shifts_middle"])):
            if (j + 1) in params["shifts_middle"]:
                if params["relative_loss"]:
                    loss3_denominator = (
                        tf.reduce_mean(
                            tf.reduce_mean(
                                tf.square(tf.squeeze(g_list[count_shifts_middle + 1])),
                                1,
                            )
                        )
                        + denominator_nonzero
                    )
                else:
                    loss3_denominator = tf.to_double(1.0)
                loss3 = loss3 + params["mid_shift_lam"] * tf.truediv(
                    tf.reduce_mean(
                        tf.reduce_mean(
                            tf.square(next_step - g_list[count_shifts_middle + 1]), 1
                        )
                    ),
                    loss3_denominator,
                )
                count_shifts_middle += 1
            omegas = KoopmanOmega.omega_net_apply.omega_net_apply(
                params, next_step, weights, biases
            )
            next_step = net.varying_multiply(
                next_step,
                omegas,
                params["delta_t"],
                params["num_real"],
                params["num_complex_pairs"],
            )

        loss3 = loss3 / params["num_shifts_middle"]

    # inf norm on autoencoder error and one prediction step
    if params["relative_loss"]:
        Linf1_den = (
            tf.norm(tf.norm(tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf)
            + denominator_nonzero
        )
        Linf2_den = (
            tf.norm(tf.norm(tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf)
            + denominator_nonzero
        )
    else:
        Linf1_den = tf.to_double(1.0)
        Linf2_den = tf.to_double(1.0)

    Linf1_penalty = tf.truediv(
        tf.norm(tf.norm(y[0] - tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf),
        Linf1_den,
    )
    Linf2_penalty = tf.truediv(
        tf.norm(tf.norm(y[1] - tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf),
        Linf2_den,
    )
    loss_Linf = params["Linf_lam"] * (Linf1_penalty + Linf2_penalty)

    loss = loss1 + loss2 + loss3 + loss_Linf

    return loss1, loss2, loss3, loss_Linf, loss


# %% Regularization of Loss Functions
def define_regularization(params, trainable_var, loss, loss1):
    """Define the regularization and add to loss.

    Arguments:
        params -- dictionary of parameters for experiment
        trainable_var -- list of trainable TensorFlow variables
        loss -- the unregularized loss
        loss1 -- the autoenocder component of the loss

    Returns:
        loss_L1 -- L1 regularization on weights W and b
        loss_L2 -- L2 regularization on weights W
        regularized_loss -- loss + regularization
        regularized_loss1 -- loss1 (autoencoder loss) + regularization

    Side effects:
        None
    """
    if params["L1_lam"]:
        l1_regularizer = tf.contrib.layers.l1_regularizer(
            scale=params["L1_lam"], scope=None
        )
        # TODO: don't include biases? use weights dict instead?
        loss_L1 = tf.contrib.layers.apply_regularization(
            l1_regularizer, weights_list=trainable_var
        )
    else:
        loss_L1 = tf.zeros(
            [
                1,
            ],
            dtype=tf.float64,
        )

    # tf.nn.l2_loss returns number
    l2_regularizer = tf.add_n(
        [tf.nn.l2_loss(v) for v in trainable_var if "b" not in v.name]
    )
    loss_L2 = params["L2_lam"] * l2_regularizer

    regularized_loss = loss + loss_L1 + loss_L2
    regularized_loss1 = loss1 + loss_L1 + loss_L2

    return loss_L1, loss_L2, regularized_loss, regularized_loss1


# %% Create Koopman Network
def KoopmanNet(data_val, params):
    """Run a random experiment for particular params and data.

    Arguments:
        data_val -- array containing validation dataset
        params -- dictionary of parameters for experiment

    Returns:
        None

    Side effects:
        Changes params dict
        Saves files
        Builds TensorFlow graph (reset in main_exp)
    """
    # setup network
    x, y, g_list, weights, biases = net.create_koopman_network(params)

    max_shifts_to_stack = hf.num_shifts_in_stack(params)

    # DEFINE LOSS FUNCTION
    trainable_var = tf.trainable_variables()
    loss1, loss2, loss3, loss_Linf, loss = define_loss(
        x, y, g_list, weights, biases, params
    )
    loss_L1, loss_L2, regularized_loss, regularized_loss1 = define_regularization(
        params, trainable_var, loss, loss1
    )

    # CHOOSE OPTIMIZATION ALGORITHM
    optimizer = hf.choose_optimizer(params, regularized_loss, trainable_var)
    optimizer_autoencoder = hf.choose_optimizer(
        params, regularized_loss1, trainable_var
    )

    # LAUNCH GRAPH AND INITIALIZE
    sess = tf.Session()
    saver = tf.train.Saver()

    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.global_variables_initializer()
    sess.run(init)

    csv_path = params["model_path"].replace("model", "error")
    csv_path = csv_path.replace("ckpt", "csv")
    print(csv_path)

    num_saved_per_file_pass = params["num_steps_per_file_pass"] / 20 + 1
    num_saved = np.floor(
        num_saved_per_file_pass
        * params["data_train_len"]
        * params["num_passes_per_file"]
    ).astype(int)
    train_val_error = np.zeros([num_saved, 16])
    count = 0
    best_error = 10000

    data_val_tensor = hf.stack_data(data_val, max_shifts_to_stack, params["len_time"])

    start = time.time()
    finished = 0
    saver.save(sess, params["model_path"])

    # TRAINING
    # loop over training data files
    for f in range(params["data_train_len"] * params["num_passes_per_file"]):
        if finished:
            break
        file_num = (f % params["data_train_len"]) + 1  # 1...data_train_len

        if (params["data_train_len"] > 1) or (f == 0):
            # don't keep reloading data if always same
            data_train = np.loadtxt(
                ("./data/%s_train%d_x.csv" % (params["data_name"], file_num)),
                delimiter=",",
                dtype=np.float64,
            )
            data_train_tensor = hf.stack_data(
                data_train, max_shifts_to_stack, params["len_time"]
            )
            num_examples = data_train_tensor.shape[1]
            num_batches = int(np.floor(num_examples / params["batch_size"]))

        ind = np.arange(num_examples)
        np.random.shuffle(ind)
        data_train_tensor = data_train_tensor[:, ind, :]

        # loop over batches in this file
        for step in range(params["num_steps_per_batch"] * num_batches):
            if params["batch_size"] < data_train_tensor.shape[1]:
                offset = (step * params["batch_size"]) % (
                    num_examples - params["batch_size"]
                )
            else:
                offset = 0

            batch_data_train = data_train_tensor[
                :, offset : (offset + params["batch_size"]), :
            ]

            feed_dict_train = {x: batch_data_train}
            feed_dict_train_loss = {x: batch_data_train}
            feed_dict_val = {x: data_val_tensor}

            if (not params["been5min"]) and params["auto_first"]:
                sess.run(optimizer_autoencoder, feed_dict=feed_dict_train)
            else:
                sess.run(optimizer, feed_dict=feed_dict_train)

            if step % 20 == 0:
                train_error = sess.run(loss, feed_dict=feed_dict_train_loss)
                val_error = sess.run(loss, feed_dict=feed_dict_val)

                if val_error < (best_error - best_error * (10 ** (-5))):
                    best_error = val_error.copy()
                    saver.save(sess, params["model_path"])
                    reg_train_err = sess.run(
                        regularized_loss, feed_dict=feed_dict_train_loss
                    )
                    reg_val_err = sess.run(regularized_loss, feed_dict=feed_dict_val)
                    print(
                        "New best val error %f (with reg. train err %f and reg. val err %f)"
                        % (best_error, reg_train_err, reg_val_err)
                    )

                train_val_error[count, 0] = train_error
                train_val_error[count, 1] = val_error
                train_val_error[count, 2] = sess.run(
                    regularized_loss, feed_dict=feed_dict_train_loss
                )
                train_val_error[count, 3] = sess.run(
                    regularized_loss, feed_dict=feed_dict_val
                )
                train_val_error[count, 4] = sess.run(
                    loss1, feed_dict=feed_dict_train_loss
                )
                train_val_error[count, 5] = sess.run(loss1, feed_dict=feed_dict_val)
                train_val_error[count, 6] = sess.run(
                    loss2, feed_dict=feed_dict_train_loss
                )
                train_val_error[count, 7] = sess.run(loss2, feed_dict=feed_dict_val)
                train_val_error[count, 8] = sess.run(
                    loss3, feed_dict=feed_dict_train_loss
                )
                train_val_error[count, 9] = sess.run(loss3, feed_dict=feed_dict_val)
                train_val_error[count, 10] = sess.run(
                    loss_Linf, feed_dict=feed_dict_train_loss
                )
                train_val_error[count, 11] = sess.run(
                    loss_Linf, feed_dict=feed_dict_val
                )

                if np.isnan(train_val_error[count, 10]):
                    params["stop_condition"] = "loss_Linf is nan"
                    finished = 1
                    break
                train_val_error[count, 12] = sess.run(
                    loss_L1, feed_dict=feed_dict_train_loss
                )
                train_val_error[count, 13] = sess.run(loss_L1, feed_dict=feed_dict_val)
                train_val_error[count, 14] = sess.run(
                    loss_L2, feed_dict=feed_dict_train_loss
                )
                train_val_error[count, 15] = sess.run(loss_L2, feed_dict=feed_dict_val)

                np.savetxt(csv_path, train_val_error, delimiter=",")
                finished, save_now = hf.check_progress(start, best_error, params)
                count = count + 1
                if save_now:
                    train_val_error_trunc = train_val_error[range(count), :]
                    hf.save_files(
                        sess, csv_path, train_val_error_trunc, params, weights, biases
                    )
                if finished:
                    break

            if step > params["num_steps_per_file_pass"]:
                params["stop_condition"] = "reached num_steps_per_file_pass"
                break

    # SAVE RESULTS
    train_val_error = train_val_error[range(count), :]
    print(train_val_error)
    params["time_exp"] = time.time() - start
    saver.restore(sess, params["model_path"])
    hf.save_files(sess, csv_path, train_val_error, params, weights, biases)
    tf.reset_default_graph()


# %% Main experiments function
def main(params):
    """Set up and run one random experiment.

    Arguments:
        params -- dictionary of parameters for experiment

    Returns:
        None

    Side effects:
        Changes params dict
        If doesn't already exist, creates folder params['folder_name']
        Saves files in that folder
    """
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
    for count in range(200):  # loop to do random experiments
        print("Experiment Number:%d" % count)
        main(params)

# %%

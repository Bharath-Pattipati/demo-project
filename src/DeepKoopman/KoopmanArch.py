# %% Import Libraries
import helperfcns as hf
import tensorflow as tf
import numpy as np


# %% Tensorflow variable for Weight matrix
def weight_variable(shape, var_name, distribution="tn", scale=0.1):
    if distribution == "tn":
        initializer = tf.keras.initializers.TruncatedNormal(stddev=scale)
    elif distribution == "xavier":
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        initializer = tf.keras.initializers.RandomUniform(-scale, scale)
    elif distribution == "dl":
        scale = 1.0 / np.sqrt(shape[0])
        initializer = tf.keras.initializers.RandomUniform(-scale, scale)
    elif distribution == "he":
        scale = np.sqrt(2.0 / shape[0])
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=scale)
    elif distribution == "glorot_bengio":
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        initializer = tf.keras.initializers.RandomUniform(-scale, scale)
    else:
        raise ValueError("Unknown distribution %s" % (distribution))

    return tf.Variable(initializer(shape=shape, dtype=tf.float64), name=var_name)


# %% Tensorflow variable for Bias vector
def bias_variable(shape, var_name, distribution=""):
    if distribution:
        return np.genfromtxt(distribution, delimiter=",", dtype=np.float64)
    else:
        return tf.constant(0.0, shape=shape, dtype=tf.float64)


# %% Encoder Network
class Encoder:
    """Create an encoder network: an input placeholder x, dictionary of weights, and dictionary of biases.

    Arguments:
        widths -- array or list of widths for layers of network
        dist_weights -- array or list of strings for distributions of weight matrices
        dist_biases -- array or list of strings for distributions of bias vectors
        scale -- (for tn distribution of weight matrices): standard deviation of normal distribution before truncation
        num_shifts_max -- number of shifts (time steps) that losses will use (max of num_shifts and num_shifts_middle)

    Returns:
        x -- placeholder for input
        weights -- dictionary of weights
        biases -- dictionary of biases

    Side effects:
        None
    """

    def __init__(self, widths, dist_weights, dist_biases, scale, num_shifts_max):
        self.widths = widths
        self.dist_weights = dist_weights
        self.dist_biases = dist_biases
        self.scale = scale
        self.num_shifts_max = num_shifts_max
        self.x = tf.Variable(
            initial_value=tf.zeros(
                [self.num_shifts_max, 1, self.widths[0]], dtype=tf.float64
            ),
            trainable=False,
        )
        self.weights = {}
        self.biases = {}
        for i in np.arange(len(widths) - 1):
            self.weights["WE" + str(i + 1)] = weight_variable(
                [self.widths[i], self.widths[i + 1]],
                "WE" + str(i + 1),
                self.dist_weights[i],
                self.scale,
            )
            self.biases["bE" + str(i + 1)] = bias_variable(
                [
                    self.widths[i + 1],
                ],
                "bE" + str(i + 1),
                self.dist_biases[i],
            )

    def apply(self, act_type, shifts_middle, name="E", num_encoder_weights=1):
        """Apply an encoder to data x.

        Arguments:
            x -- input tensor
            weights -- dictionary of weights
            biases -- dictionary of biases
            act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
            shifts_middle -- number of shifts (steps) in x to apply encoder to for linearity loss
            name -- string for prefix on weight matrices (default 'E' for encoder)
            num_encoder_weights -- number of weight matrices (layers) in encoder network (default 1)

        Returns:
            y -- list, output of encoder network applied to each time shift in input x

        Side effects:
            None
        """
        self.y = []
        num_shifts_middle = len(shifts_middle)
        for j in range(num_shifts_middle):
            if j == 0:
                shift = 0
            else:
                shift = shifts_middle[j - 1]
            x_shift = tf.squeeze(self.x[shift, :, :])
            self.y.append(
                self.apply_one_shift(
                    x_shift,
                    self.weights,
                    self.biases,
                    act_type,
                    name,
                    num_encoder_weights,
                )
            )

    @staticmethod
    def apply_one_shift(
        prev_layer, weights, biases, act_type, name="E", num_encoder_weights=1
    ):
        """Apply an encoder to data for only one time step (shift).

        Arguments:
            prev_layer -- input for a particular time step (shift)
            weights -- dictionary of weights
            biases -- dictionary of biases
            act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
            name -- string for prefix on weight matrices (default 'E' for encoder)
            num_encoder_weights -- number of weight matrices (layers) in encoder network (default 1)

        Returns:
            final -- output of encoder network applied to input prev_layer (a particular time step / shift)

        Side effects:
            None
        """

        for i in range(num_encoder_weights - 1):
            # Reshape prev_layer to be 2-dimensional
            prev_layer = tf.reshape(
                prev_layer, [-1, weights["W%s%d" % (name, i + 1)].shape[0]]
            )

            prev_layer = (
                tf.matmul(prev_layer, weights["W%s%d" % (name, i + 1)])
                + biases["b%s%d" % (name, i + 1)]
            )
            if act_type == "sigmoid":
                prev_layer = tf.sigmoid(prev_layer)
            elif act_type == "relu":
                prev_layer = tf.nn.relu(prev_layer)
            elif act_type == "elu":
                prev_layer = tf.nn.elu(prev_layer)

        # apply last layer without any nonlinearity
        prev_layer = tf.reshape(
            prev_layer, [-1, weights["W%s%d" % (name, num_encoder_weights)].shape[0]]
        )

        final = (
            tf.matmul(prev_layer, weights["W%s%d" % (name, num_encoder_weights)])
            + biases["b%s%d" % (name, num_encoder_weights)]
        )

        return final


# %% Omega Network
class Omega:
    """Create the auxiliary (omega) network(s), which have ycoords as input and output omegas (parameters for L).

    Arguments:
        params -- dictionary of parameters for experiment
        ycoords -- array of shape [None, k] of y-coordinates, where L will be k x k

    Returns:
        omegas -- list, output of omega (auxiliary) network(s) applied to input ycoords
        weights -- dictionary of weights
        biases -- dictionary of biases

    Side effects:
        Adds 'num_omega_weights' key to params dict
    """

    def __init__(self, params, ycoords):
        self.weights = {}
        self.biases = {}

        for j in np.arange(params["num_complex_pairs"]):
            temp_name = "OC%d_" % (j + 1)
            self.create_one_omega_net(
                params,
                temp_name,
                params["widths_omega_complex"],
            )
        for j in np.arange(params["num_real"]):
            temp_name = "OR%d_" % (j + 1)
            self.create_one_omega_net(
                params,
                temp_name,
                params["widths_omega_real"],
            )

        params["num_omega_weights"] = len(params["widths_omega_real"]) - 1

        self.omegas = self.omega_net_apply(params, ycoords, self.weights, self.biases)

    def create_one_omega_net(self, params, name, widths):
        """Create one auxiliary (omega) network for one real eigenvalue or a pair of complex conj. eigenvalues.

        Arguments:
            params -- dictionary of parameters for experiment
            temp_name -- string for prefix on weight matrices, i.e. OC1 or OR1
            weights -- dictionary of weights
            biases -- dictionary of biases
            widths -- array or list of widths for layers of network

        Returns:
            None

        Side effects:
            Updates weights and biases dictionaries
        """
        OmegaDecoder = Decoder(
            widths,
            dist_weights=params["dist_weights_omega"],
            dist_biases=params["dist_biases_omega"],
            scale=params["scale_omega"],
            name=name,
        )

        weightsO = OmegaDecoder.weights
        biasesO = OmegaDecoder.biases
        self.weights.update(weightsO)
        self.biases.update(biasesO)

    def omega_net_apply(self, params, ycoords, weights, biases):
        """Apply the omega (auxiliary) network(s) to the y-coordinates.

        Arguments:
            params -- dictionary of parameters for experiment
            ycoords -- array of shape [None, k] of y-coordinates, where L will be k x k
            weights -- dictionary of weights
            biases -- dictionary of biases

        Returns:
            omegas -- list, output of omega (auxiliary) network(s) applied to input ycoords

        Side effects:
            None
        """
        omegas = []
        for j in np.arange(params["num_complex_pairs"]):
            temp_name = "OC%d_" % (j + 1)
            ind = 2 * j
            pair_of_columns = ycoords[:, ind : ind + 2]
            radius_of_pair = tf.reduce_sum(
                tf.square(pair_of_columns), axis=1, keep_dims=True
            )
            omegas.append(
                self.apply_one(params, radius_of_pair, weights, biases, temp_name)
            )
        for j in np.arange(params["num_real"]):
            temp_name = "OR%d_" % (j + 1)
            ind = 2 * params["num_complex_pairs"] + j
            one_column = ycoords[:, ind]
            omegas.append(
                self.apply_one(
                    params, one_column[:, np.newaxis], weights, biases, temp_name
                )
            )

        return omegas

    def apply_one(self, params, ycoords, weights, biases, name):
        """Apply one auxiliary (omega) network for one real eigenvalue or a pair of complex conj. eigenvalues.

        Arguments:
            params -- dictionary of parameters for experiment
            ycoords -- array of shape [None, k] of y-coordinates, where L will be k x k
            weights -- dictionary of weights
            biases -- dictionary of biases
            name -- string for prefix on weight matrices, i.e. OC1 or OR1

        Returns:
            omegas - output of one auxiliary (omega) network to input ycoords

        Side effects:
            None
        """
        omegas = Encoder.apply_one_shift(
            ycoords,
            weights,
            biases,
            params["act_type"],
            name=name,
            num_encoder_weights=params["num_omega_weights"],
        )
        return omegas


# %% Decoder Network
class Decoder:
    """Create a decoder network: a dictionary of weights and a dictionary of biases.

    Arguments:
        widths -- array or list of widths for layers of network
        dist_weights -- array or list of strings for distributions of weight matrices
        dist_biases -- array or list of strings for distributions of bias vectors
        scale -- (for tn distribution of weight matrices): standard deviation of normal distribution before truncation
        name -- string for prefix on weight matrices (default 'D' for decoder)

    Returns:
        weights -- dictionary of weights
        biases -- dictionary of biases

    Side effects:
        None
    """

    def __init__(self, widths, dist_weights, dist_biases, scale, name="D"):
        self.weights = {}
        self.biases = {}
        for i in np.arange(len(widths) - 1):
            ind = i + 1
            self.weights["W%s%d" % (name, ind)] = weight_variable(
                [widths[i], widths[i + 1]],
                var_name="W%s%d" % (name, ind),
                distribution=dist_weights[ind - 1],
                scale=scale,
            )
            self.biases["b%s%d" % (name, ind)] = bias_variable(
                [
                    widths[i + 1],
                ],
                var_name="b%s%d" % (name, ind),
                distribution=dist_biases[ind - 1],
            )

    def apply(self, prev_layer, weights, biases, act_type, num_decoder_weights):
        """Apply a decoder to data prev_layer

        Arguments:
            prev_layer -- input to decoder network
            weights -- dictionary of weights
            biases -- dictionary of biases
            act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
            num_decoder_weights -- number of weight matrices (layers) in decoder network

        Returns:
            output of decoder network applied to input prev_layer

        Side effects:
            None
        """
        for i in np.arange(num_decoder_weights - 1):
            prev_layer = (
                tf.matmul(prev_layer, weights["WD%d" % (i + 1)])
                + biases["bD%d" % (i + 1)]
            )
            if act_type == "sigmoid":
                prev_layer = tf.sigmoid(prev_layer)
            elif act_type == "relu":
                prev_layer = tf.nn.relu(prev_layer)
            elif act_type == "elu":
                prev_layer = tf.nn.elu(prev_layer)

        # apply last layer without any nonlinearity
        return (
            tf.matmul(prev_layer, weights["WD%d" % num_decoder_weights])
            + biases["bD%d" % num_decoder_weights]
        )


# %% Miscellaneous Functions
def form_complex_conjugate_block(omegas, delta_t):
    """Form a 2x2 block for a complex conj. pair of eigenvalues, but for each example, so dimension [None, 2, 2]

    2x2 Block is
    exp(mu * delta_t) * [cos(omega * delta_t), -sin(omega * delta_t)
                         sin(omega * delta_t), cos(omega * delta_t)]

    Arguments:
        omegas -- array of parameters for blocks. first column is freq. (omega) and 2nd is scaling (mu), size [None, 2]
        delta_t -- time step in trajectories from input data

    Returns:
        stack of 2x2 blocks, size [None, 2, 2], where first dimension matches first dimension of omegas

    Side effects:
        None
    """
    scale = tf.exp(omegas[:, 1] * delta_t)
    entry11 = tf.multiply(scale, tf.cos(omegas[:, 0] * delta_t))
    entry12 = tf.multiply(scale, tf.sin(omegas[:, 0] * delta_t))
    row1 = tf.stack([entry11, -entry12], axis=1)  # [None, 2]
    row2 = tf.stack([entry12, entry11], axis=1)  # [None, 2]
    return tf.stack([row1, row2], axis=2)  # [None, 2, 2] put one row below other


def varying_multiply(y, omegas, delta_t, num_real, num_complex_pairs):
    """Multiply y-coordinates on the left by matrix L, but let matrix vary.

    Arguments:
        y -- array of shape [None, k] of y-coordinates, where L will be k x k
        omegas -- list of arrays of parameters for the L matrices
        delta_t -- time step in trajectories from input data
        num_real -- number of real eigenvalues
        num_complex_pairs -- number of pairs of complex conjugate eigenvalues

    Returns:
        array same size as input y, but advanced to next time step

    Side effects:
        None
    """
    complex_list = []

    # first, Jordan blocks for each pair of complex conjugate eigenvalues
    for j in np.arange(num_complex_pairs):
        ind = 2 * j
        ystack = tf.stack(
            [y[:, ind : ind + 2], y[:, ind : ind + 2]], axis=2
        )  # [None, 2, 2]
        L_stack = form_complex_conjugate_block(omegas[j], delta_t)
        elmtwise_prod = tf.multiply(ystack, L_stack)
        complex_list.append(tf.reduce_sum(elmtwise_prod, 1))

    if len(complex_list):
        # each element in list output_list is shape [None, 2]
        complex_part = tf.concat(complex_list, axis=1)

    # next, diagonal structure for each real eigenvalue
    # faster to not explicitly create stack of diagonal matrices L
    real_list = []
    for j in np.arange(num_real):
        ind = 2 * num_complex_pairs + j
        temp = y[:, ind]
        real_list.append(
            tf.multiply(
                temp[:, np.newaxis], tf.exp(omegas[num_complex_pairs + j] * delta_t)
            )
        )

    if len(real_list):
        real_part = tf.concat(real_list, axis=1)
    if len(complex_list) and len(real_list):
        return tf.concat([complex_part, real_part], axis=1)
    elif len(complex_list):
        return complex_part
    else:
        return real_part


# %% Main Koopman Network
def create_koopman_network(params):
    """Create a Koopman network that encodes, advances in time, and decodes.

    Arguments:
        params -- dictionary of parameters for experiment

    Returns:
        x -- placeholder for input
        y -- list, output of decoder applied to each shift: g_list[0], K*g_list[0], K^2*g_list[0], ..., length num_shifts + 1
        g_list -- list, output of encoder applied to each shift in input x, length num_shifts_middle + 1
        weights -- dictionary of weights
        biases -- dictionary of biases

    Side effects:
        Adds more entries to params dict: num_encoder_weights, num_omega_weights, num_decoder_weights

    Raises ValueError if len(y) is not len(params['shifts']) + 1
    """
    ### Encoder ###
    depth = int((params["d"] - 4) / 2)
    max_shifts_to_stack = hf.num_shifts_in_stack(params)
    encoder_widths = params["widths"][0 : depth + 2]  # n ... k

    KoopmanEncoder = Encoder(
        encoder_widths,
        dist_weights=params["dist_weights"][0 : depth + 1],
        dist_biases=params["dist_biases"][0 : depth + 1],
        scale=params["scale"],
        num_shifts_max=max_shifts_to_stack,
    )

    params["num_encoder_weights"] = len(KoopmanEncoder.weights)

    KoopmanEncoder.apply(
        params["act_type"],
        shifts_middle=params["shifts_middle"],
        num_encoder_weights=params["num_encoder_weights"],
    )

    x = KoopmanEncoder.x
    g_list = KoopmanEncoder.y
    weights = KoopmanEncoder.weights
    biases = KoopmanEncoder.biases

    ### Omega: Auxiliary ###
    # params['num_omega_weights'] = len(weights_omega) already done inside create_omega_net
    KoopmanOmega = Omega(params, g_list[0])
    omegas = KoopmanOmega.omegas
    weights.update(KoopmanOmega.weights)
    biases.update(KoopmanOmega.biases)

    ### Decoder ###
    num_widths = len(params["widths"])
    decoder_widths = params["widths"][depth + 2 : num_widths]  # k ... n
    KoopmanDecoder = Decoder(
        decoder_widths,
        dist_weights=params["dist_weights"][depth + 2 :],
        dist_biases=params["dist_biases"][depth + 2 :],
        scale=params["scale"],
    )
    weights_decoder = KoopmanDecoder.weights
    biases_decoder = KoopmanDecoder.biases
    weights.update(weights_decoder)
    biases.update(biases_decoder)

    y = []  # Encoded variables
    # y[0] is x[0,:,:] encoded and then decoded (no stepping forward)
    encoded_layer = g_list[0]
    params["num_decoder_weights"] = depth + 1
    y.append(
        KoopmanDecoder.apply(
            encoded_layer,
            weights,
            biases,
            params["act_type"],
            params["num_decoder_weights"],
        )
    )

    # g_list_omega[0] is for x[0,:,:], pairs with g_list[0]=encoded_layer
    advanced_layer = varying_multiply(
        encoded_layer,
        omegas,
        params["delta_t"],
        params["num_real"],
        params["num_complex_pairs"],
    )

    for j in np.arange(max(params["shifts"])):
        # considering penalty on subset of yk+1, yk+2, yk+3, ...
        if (j + 1) in params["shifts"]:
            y.append(
                KoopmanDecoder.apply(
                    advanced_layer,
                    weights,
                    biases,
                    params["act_type"],
                    params["num_decoder_weights"],
                )
            )

        omegas = KoopmanOmega.omega_net_apply(params, advanced_layer, weights, biases)
        advanced_layer = varying_multiply(
            advanced_layer,
            omegas,
            params["delta_t"],
            params["num_real"],
            params["num_complex_pairs"],
        )

    if len(y) != (len(params["shifts"]) + 1):
        print("messed up looping over shifts! %r" % params["shifts"])
        raise ValueError(
            "length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment"
        )

    return x, y, g_list, weights, biases

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

    def apply_one_shift(
        self, prev_layer, weights, biases, act_type, name="E", num_encoder_weights=1
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


# %% Main Koopman Network
def create_koopman_network(params):
    """
    Create a Koopman network based on the given parameters.
    """
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

    return KoopmanEncoder

# %% Import Libraries
import torch
from torch import nn, optim

# import random
# import matplotlib.pyplot as plt
# import numpy as np

from torchvision.models import resnet18, ResNet18_Weights


# %% Basic
""" x = torch.rand(5, 3)
print(x) """

# %% Andrej Karpathy Tutorial
# The spelled-out intro to neural networks and backpropagation: building micrograds
""" x1 = torch.Tensor([2.0]).double()
x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()
x2.requires_grad = True

w1 = torch.Tensor([-3.0]).double()
w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()
w2.requires_grad = True

b = torch.Tensor([6.8813735870195432]).double()
n = x1 * w1 + x2 * w2 + b
y = torch.tanh(n)

# Backprop i.e. Chain Rule
y.backward()  # compute gradients of a tensor w.r.t computational graph leaves
print(f"y: {y.data.item()}")  # return the element, stripping out Tensor wrapper
print(f"x1.grad: {x1.grad.item()}")
print(f"x2.grad: {x2.grad.item()}")
print(f"w1.grad: {w1.grad.item()}")
print(f"w2.grad: {w2.grad.item()}") """


# %% Single Neuron & Layer of Neurons
""" class Neuron:
    def __init__(self, nin):
        self.w = [random.uniform(-1, 1) for _ in range(nin)]
        self.b = random.uniform(-1, 1)

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = np.tanh(act).item()
        return out


class Layer:
    # nin: Number of inputs
    # nout: Number of neurons in a single layer
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs


class MLP:
    """
"""     Constructor for MLP class.

    Parameters
    ----------
    nin : int
        Number of input neurons.
    nouts : list of int
        A list of the number of neurons in each layer of the MLP.

    Returns
    -------
    self : MLP object
        An instance of the MLP class. """

"""     def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def testClasses():
    x = [2.0, 3.0]
    n = Neuron(2)
    print(f"Neuron: {n(x)}")

    n = Layer(2, 3)  # 2 dimensional neurons and 3 of them
    print(f"Layer: {n(x)}")

    x = [2.0, 3.0, -1.0]  # 3 inputs
    n = MLP(
        3, [4, 4, 1]
    )  # 3 inputs, 4 neurons in first layer, 4 in second and 1 output
    print(f"MLP: {n(x)}")

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # Desired Targets
    ypred = [n(x) for x in xs]

    loss = sum((y - ypred) ** 2 for y, ypred in zip(ys, ypred))

    print(f"ypred: {ypred}")
    print(f"Loss: {loss}") """

""" 
if __name__ == "__main__":
    testClasses() 
"""

# %% Torch Implementation

""" class Module:
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin):
        self.w = 2 * torch.rand(nin) - 1
        self.w.requires_grad = True
        self.b = torch.rand(1)
        self.b.requires_grad = True

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = torch.tanh(act)
        return out

    def parameters(self):
        return torch.cat([self.w, self.b]) """


""" class Layer(Module):
    # nin: Number of inputs
    # nout: Number of neurons in a single layer
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()] """


""" class MLP(Module):
    """

"""     def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]  """


""" def testClasses():
    x = [2.0, 3.0]
    n = Neuron(2)
    print(f"Neuron: {n(x)}")

    n = Layer(2, 3)  # 2 dimensional neurons and 3 of them
    print(f"Layer: {n(x)}")

    x = [2.0, 3.0, -1.0]  # 3 inputs
    n = MLP(
        3, [4, 4, 1]
    )  # 3 inputs, 4 neurons in first layer, 4 in second and 1 output
    print(f"MLP: {n(x)}")
    print(f"# of parameters: {len(n.parameters())}")

    xs = torch.Tensor(
        [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
    ).double()
    xs.requires_grad = True
    ys = torch.Tensor([1.0, -1.0, -1.0, 1.0]).double()  # Desired Targets

    for k in range(10):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((y - ypred) ** 2 for y, ypred in zip(ys, ypred))

        # backward pass
        n.zero_grad()
        loss.backward()

        # update the weights
        for p in n.parameters():
            if p.grad is not None:  # Check if gradient is not None
                p.data -= 0.01 * p.grad  # minimize the loss
            else:
                print(f"Gradient is None for parameter {p}")

        print(f"Epoch: {k}, Loss: {loss.data.item()}") """

# print(f"ypred: {ypred}")
# print(f"Loss: {loss}")
# print(f"Gradient: {n.layers[0].neurons[0].w.grad[0]}")
# print(f"Data: {n.layers[0].neurons[0].w.data[0]}")


""" if __name__ == "__main__":
    testClasses() """

# %% Tensors
""" data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# Shape is a tuple of tensor dimensions
shape = (2, 3)
rand_tensor = torch.rand(shape)
print(f"Random Tensor: \n {rand_tensor} \n")

tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(f"Device tensor is stored on: {tensor.device}")

# joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# matrix multiplication between 2 tensors
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# numpy to tensor
n = np.ones(5)
t = torch.from_numpy(n)

# change in numpy array reflects in tensor
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}") """

# %% AUTOGRAD
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 1000)

data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# forward pass
predictions = model(data)
loss = (predictions - labels).sum()

# backward pass: computes gradient of the loss with respect to model parameters
loss.backward()

# update weights
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer.step()

# %%

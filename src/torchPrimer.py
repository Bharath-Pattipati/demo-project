# %% Import Libraries
import torch
# import math

# import torchvision
# import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

# import random
# import matplotlib.pyplot as plt
# import numpy as np

# from torchvision.models import resnet18, ResNet18_Weights


# %% Basic
"""x = torch.rand(5, 3)
print(x)"""

""" x = [[5, 3], [0, 9]]
y = torch.tensor(x)
print(y) """

""" # random numbers from uniform distribution on interval [0, 1)
x = torch.rand(4, 4)
y = torch.rand(4, 4)
print(x)
print(x.device)

z = torch.vstack([x, y])
print(z)

# tensor of shape (2,4,4)
z = torch.stack([x, y], dim=2)
print(z)

print(z[3,3,0]) """

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
""" model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 1000)

data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# forward pass
predictions = model(data)
loss = (predictions - labels).sum()
print(loss.grad_fn)
print(model.conv1.weight.grad)

# backward pass: computes gradient of the loss with respect to model parameters
loss.backward()

# update weights
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer.step() """


# %% Neural Networks
""" class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # call the parent class constructor
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(
            1, 6, 5
        )  # nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # define the forward function, and the backward function (where gradients are computed) is automatically defined for you using autograd
    def forward(self, input):
        # Convolution layer C1: 1 input channel, 6 output channels, 5x5 square convolution kernel with RELU activation and outputs a tensor of size (N, 6, 28, 28), where N is size of the batch
        c1 = F.relu(self.conv1(input))
        # Subsampling layer S2: 2x2 max pooling layer with stride 2, purely functional, and outputs a tensor of size (N, 6, 14, 14)
        s2 = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels, 5x5 square convolution kernel with RELU activation and outputs a tensor of size (N, 16, 10, 10)
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 max pooling layer with stride 2, purely functional, and outputs a tensor of size (N, 16, 5, 5)
        s4 = F.max_pool2d(c3, 2)
        # Flattern operation: purefly functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 400) Tensor input, and outputs a (N, 120) Tensor
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input, and outputs a (N, 84) Tensor
        f6 = F.relu(self.fc2(f5))
        # Gaussian layer output: (N, 84) Tensor input, and outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output


net = Net()
print(net)

# params = list(net.parameters())
# print(f"Number of parameters: {len(params)}")
# print(params[0].size())  # conv1's .weight

# Expected input size of LeNet is 32x32
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# zero the gradient buffers of all parameters and backprops with random gradients
net.zero_grad()
out.backward(
    torch.randn(1, 10)
)  # whole graph is differentiated w.r.t. the neural net parameters

# Loss Function
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# few steps backward
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# Backpropagation: clear existing gradients, else gradients will be accumulated to existing gradients
net.zero_grad()  # zeroes the gradient buffers of all parameters

print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)

loss.backward()

print("conv1.bias.grad after backward")
print(net.conv1.bias.grad)

# Update weights using Stochastic Gradient Descent (SGD)
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# params before optimization
params = list(net.parameters())
print(f"parameters: \n{params}")

# Different update rules from torch.optim (SGD, Nesterov-SGD, Adam, RMSProp, etc.)
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()  # zero the gradient buffers
output = net(input)  # forward pass
loss = criterion(output, target)  # compute loss
loss.backward()  # backward pass
optimizer.step()  # Update weights """


# %% Simple Neural Net
""" class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
predictions = net.forward(torch.randn(1, 1, 32, 32))
print(len(list(net.parameters())))

# names of net parameters
for name, param in net.named_parameters():
    print(name, param.size()) """

# %% Training Classifier CIFAR-10 dataset [size: 3x32x32]
# For vision, we have created a package called torchvision, that has data loaders for common datasets such as ImageNet, CIFAR10, MNIST, etc.
# and data transformers for images, viz., torchvision.datasets and torch.utils.data.DataLoader.

""" if __name__ == "__main__":
    # Load and normalize CIFAR10 training and test datasets using torchvision
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="../data/processed/", train=True, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="../data/processed/", train=False, download=True, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

    # Define CNN
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    # Define Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train network on training data
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")

    # save trained model
    PATH = "../models/cifar_net.pth"
    torch.save(net.state_dict(), PATH)

    # Test network on test data
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print("Predicted: ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
    )

    # what classes performed well and the classes that did not perform well
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                classname = classes[label]
                correct_pred[classname] += c[i].item()
                total_pred[classname] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * correct_count / total_pred[classname]
        print(f"Accuracy for class {classname:5s} is: {accuracy:.1f} %")

    # Training o GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # move model and data to GPU
    net.to(device)
    inputs, labels = inputs.to(device), labels.to(device) """

# %% NN Package: Polynomial Regression
""" 
# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
# (3,), for this case, broadcasting semantics will apply to obtain a tensor
# of shape (2000, 3)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. The Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# The Flatten layer flatens the output of the linear layer to a 1D tensor,
# to match the shape of `y`.
model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(xx)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# You can access the first layer of `model` like accessing the first item of a list
linear_layer = model[0]

# For linear layer, its parameters are stored as `weight` and `bias`.
print(
    f"Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3"
) """


# %% Self-Attention Class
class SelfAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        ## d_model = the number of embedding values per token.
        ##           Because we want to be able to do the math by hand, we've
        ##           the default value for d_model=2.
        ##           However, in "Attention Is All You Need" d_model=512
        ##
        ## row_dim, col_dim = the indices we should use to access rows or columns
        super().__init__()

        ## Initialize the Weights (W) that we'll use to create the
        ## query (q), key (k) and value (v) for each token
        ## NOTE: A lot of implementations include bias terms when
        ##       creating the queries, keys, and values, but
        ##       the original manuscript that described Attention,
        ##       "Attention Is All You Need" did not, so we won't either
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encodings):
        ## Create the query, key and values using the encoding numbers
        ## associated with each token (token encodings)
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        # Compute similarity scores: (q * k^T)
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        # Scalre the similarities by dividing by the square root of the dimension of the key vectors
        scaled_sims = sims / torch.tensor(k.size(self.col_dim) ** 0.5)

        # Apply softmax to the similarity scores to determine what percent of each tokens value to use in final attention values.
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        # Scale the values by their associated percentages and add them up
        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores


# %% Calculate Self-Attention
## create a matrix of token encodings...
encodings_matrix = torch.tensor([[1.16, 0.23], [0.57, 1.36], [4.41, -2.16]])

## set the seed for the random number generator
torch.manual_seed(42)

## create a basic self-attention ojbect
selfAttention = SelfAttention(d_model=2, row_dim=0, col_dim=1)

## calculate basic attention for the token encodings
selfAttention(encodings_matrix)

# %% Verify Calculations
## print out the weight matrix that creates the queries
selfAttention.W_q.weight.transpose(0, 1)

## print out the weight matrix that creates the keys
selfAttention.W_k.weight.transpose(0, 1)

## print out the weight matrix that creates the values
selfAttention.W_v.weight.transpose(0, 1)

## calculate the queries
selfAttention.W_q(encodings_matrix)

## calculate the keys
selfAttention.W_k(encodings_matrix)

## calculate the values
selfAttention.W_v(encodings_matrix)

q = selfAttention.W_q(encodings_matrix)
q

k = selfAttention.W_k(encodings_matrix)
k

sims = torch.matmul(q, k.transpose(dim0=0, dim1=1))
sims

scaled_sims = sims / (torch.tensor(2) ** 0.5)
scaled_sims

attention_percents = F.softmax(scaled_sims, dim=1)
attention_percents

torch.matmul(attention_percents, selfAttention.W_v(encodings_matrix))

# %%

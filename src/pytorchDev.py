import torch

x = torch.rand(5, 3)
print(x)

# %% Andrej Karpathy Tutorial
# The spelled-out intro to neural networks and backpropagation: building micrograds
x1 = torch.Tensor([2.0]).double()
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
print(f"y: {y.data.item()}")
print(f"x1.grad: {x1.grad}")
print(f"x2.grad: {x2.grad}")
print(f"w1.grad: {w1.grad}")
print(f"w2.grad: {w2.grad}")

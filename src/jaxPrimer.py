# %% Import libraries
import jax.numpy as jnp
from jax import random
from jax import jit
from jax import grad
from jax import jacobian
from jax import jacfwd, jacrev
from jax import vmap

import timeit
import numpy as np


# %% Quickstart
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


x = jnp.arange(5.0)
print(selu(x))

# %% Just-In-Time (JIT) compilation
key = random.key(1701)
x = random.normal(key, (1_000_000,))

# Use timeit module to measure execution time
result = timeit.repeat(lambda: selu(x).block_until_ready(), number=100, repeat=7)
mean_time = np.mean(result)
std_dev = np.std(result)
print(
    f"Execution time: {mean_time * 1000:.2f} ms ± {std_dev * 1000:.2f} μs per loop (mean ± std. dev. of {len(result)} runs, {100} loops each)"
)

selu_jit = jit(selu)
_ = selu_jit(x)  # compiles on first call
result = timeit.repeat(lambda: selu(x).block_until_ready(), number=100, repeat=7)
mean_time = np.mean(result)
std_dev = np.std(result)
print(
    f"Execution time: {mean_time * 1000:.2f} ms ± {std_dev * 1000:.2f} μs per loop (mean ± std. dev. of {len(result)} runs, {100} loops each)"
)


# %% Taking derivatives with jax grad
def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))


x_small = jnp.arange(3.0)
derivative_fn = grad(sum_logistic)
print("JAX GRAD: \n", derivative_fn(x_small))


# finite difference results
def first_finite_differences(f, x, eps=1e-3):
    return jnp.array(
        [(f(x + eps * v) - f(x - eps * v)) / (2 * eps) for v in jnp.eye(len(x))]
    )


print("Finite difference: \n", first_finite_differences(sum_logistic, x_small))

print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))
# %% Jacobian Matrix
# Compute Jacobian Matrix for vector-valued functions

print(jacobian(jnp.exp)(x_small))


# %% Hessian
def hessian(fun):
    return jit(jacfwd(jacrev(fun)))


print(hessian(sum_logistic)(x_small))

# Advanced Auto-diff: jax.vjp() for reverse-mode vector-Jacobian products and jax.jvp() for forward-mode Jacobian-vector products

# %% Auto-vectorization with jax.vmap()
key1, key2 = random.split(key)
mat = random.normal(key1, (150, 100))
batched_x = random.normal(key2, (10, 100))


def apply_matrix(x):
    return jnp.dot(mat, x)


def naively_batched_apply_matrix(v_batched):
    return jnp.stack([apply_matrix(v) for v in v_batched])


print("Naively batched:")
result = timeit.repeat(
    lambda: naively_batched_apply_matrix(batched_x).block_until_ready(),
    number=100,
    repeat=7,
)
mean_time = np.mean(result)
std_dev = np.std(result)
print(
    f"Execution time: {mean_time * 1000:.2f} ms ± {std_dev * 1000:.2f} μs per loop (mean ± std. dev. of {len(result)} runs, {100} loops each)"
)


@jit
def batched_apply_matrix(batched_x):
    return jnp.dot(batched_x, mat.T)


np.testing.assert_allclose(
    naively_batched_apply_matrix(batched_x),
    batched_apply_matrix(batched_x),
    atol=1e-4,
    rtol=1e-4,
)
print("Manually batched:")
result = timeit.repeat(
    lambda: batched_apply_matrix(batched_x).block_until_ready(),
    number=100,
    repeat=7,
)
mean_time = np.mean(result)
std_dev = np.std(result)
print(
    f"Execution time: {mean_time * 1000:.2f} ms ± {std_dev * 1000:.2f} μs per loop (mean ± std. dev. of {len(result)} runs, {100} loops each)"
)


@jit
def vmap_batched_apply_matrix(batched_x):
    return vmap(apply_matrix)(batched_x)


np.testing.assert_allclose(
    naively_batched_apply_matrix(batched_x),
    vmap_batched_apply_matrix(batched_x),
    atol=1e-4,
    rtol=1e-4,
)
print("Auto-vectorized with vmap:")
result = timeit.repeat(
    lambda: vmap_batched_apply_matrix(batched_x).block_until_ready(),
    number=100,
    repeat=7,
)
mean_time = np.mean(result)
std_dev = np.std(result)
print(
    f"Execution time: {mean_time * 1000:.2f} ms ± {std_dev * 1000:.2f} μs per loop (mean ± std. dev. of {len(result)} runs, {100} loops each)"
)
# %%

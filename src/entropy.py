import numpy as np

import jax.numpy as jnp
from jax import grad, jit, random, value_and_grad, vmap
from jax.scipy.special import logsumexp as lse

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

import matplotlib.pyplot as plt
import seaborn as sns

seed = 1111
key = random.PRNGKey(seed)

# number of dots
N = 3
# number of features
K = 5

key, W_key = random.split(key)
W = random.poisson(W_key, 1, shape=(K, N))

def logistic_regression(W, x):
    score = x @ W
    return score - lse(score)

def entropy(W, x):
    return tfd.Categorical(logits=logistic_regression(W, x)).entropy()

print(logistic_regression(W, jnp.zeros((K,))))
print(entropy(W, jnp.zeros((K,))))

all_3 = [
    (x, y, z ) for x in range(K) for y in range(x+1, K) for z in range(y+1, K)
]
all_3_vectors = jnp.array(all_3)

x = jnp.zeros((all_3_vectors.shape[0], K)).at[
    jnp.arange(all_3_vectors.shape[0])[:,None], all_3_vectors].set(1)
H1 = entropy(W, x)
print(f"H1.argmin {H1.argmin().item()}")

# choose lowest entropy x
idxs = all_3_vectors[H1.argmin()]
# add that to x and check resulting entropy
x = x.at[:, idxs].add(1)
H2 = entropy(W, x)
print(f"H2.argmin {H2.argmin().item()}")

# stacked bar chart
import pandas as pd
df = pd.DataFrame({
    "First turn": H1,
    "Second turn": H2,
}, index = np.arange(len(all_3)))

df.plot(kind="bar", stacked=False, color=["blue", "green"])
plt.xlabel("Action")
plt.ylabel("Entropy")
plt.savefig("entropy.png")


import jax
import jax.numpy as jnp


class FakeModel:
    """
    A model where parameters control the mean and covariance of a multivariate normal.
    - The first num_outputs parameters are the means.
    - The remaining parameters parameterize the lower-triangular matrix L of the Cholesky decomposition of the covariance matrix.
      The diagonal entries of L are parameterized as exp(x) to ensure positivity.
    """

    def __init__(self, num_outputs):
        self.num_outputs = num_outputs
        self.num_params = num_outputs * (num_outputs + 3) // 2

    def unpack_params(self, params):
        means = params[:self.num_outputs]
        L = jnp.zeros((self.num_outputs, self.num_outputs))
        idx = self.num_outputs
        for i in range(self.num_outputs):
            for j in range(i + 1):
                if i == j:
                    # Diagonal: exp to ensure positive
                    L = L.at[i, j].set(jnp.exp(params[idx]))
                else:
                    # Off-diagonal: unconstrained
                    L = L.at[i, j].set(params[idx])
                idx += 1
        return means, L

    def compute_outputs(self, params=None, n_samples=1000, key=None):
        if params is None:
            params = jnp.zeros(self.num_params)
        means, L = self.unpack_params(params)
        if key is None:
            key = jax.random.PRNGKey(0)
        z = jax.random.normal(key, (n_samples, self.num_outputs))
        samples = means + z @ L.T

        weights = jnp.ones(n_samples)
        # Downweight samples whose i-band magnitude is far below -2
        weights /= 1 + jnp.exp(-2 - samples[:, 0])

        return samples, weights

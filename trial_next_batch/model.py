"""Small grouped Gaussian-process model used by the planning step.

The model deliberately keeps the two responses separate.  It does not combine
strength and loss into a single score; each response gets its own posterior and
its own uncertainty.  Trials from the same furnace batch share a random effect,
represented as a block in the noise covariance.  Pending and failed trials are
not passed to this model at all.

A linear trend is included as a mean function because industrial response data
often has a dominant first-order trend and a pure constant mean would shrink
such trends away from the data.  The trend coefficients are profiled out
analytically, so no regularization weight needs to be chosen by hand.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


# Variance floor in standardized units.  Square of roughly 2% of the observed
# spread in each response: enough to keep intervals from claiming impossible
# precision, small enough not to distort real differences.
NOISE_FLOOR = 4e-4


class ModelError(ValueError):
    """Raised when the available observations cannot support a stable fit."""


def _kernel(x, z, length_scales, signal_variance, constant):
    scaled = (x[:, None, :] - z[None, :, :]) / length_scales
    squared = np.sum(scaled * scaled, axis=2)
    return signal_variance * np.exp(-0.5 * squared) + constant


class GroupedGP:
    """Gaussian process with a linear trend and a per-batch random intercept.

    ``y_i = h(x_i) beta + f(x_i) + b_{batch(i)} + e_i`` with ``h(x) = [1, x]``.
    All trials of one batch share ``b``, so the covariance has a dense block per
    batch.  The consequence is that a new trial must carry the same batch
    variance as a new furnace, and that repeated trials from one batch do not
    count as independent evidence.
    """

    def __init__(self, x, y, groups):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.groups = np.asarray(groups)
        if self.x.ndim != 2 or len(self.x) != len(self.y) or len(self.y) != len(self.groups):
            raise ModelError('模型输入维度不一致')
        if len(self.y) < 2:
            raise ModelError('观测不足，无法拟合响应模型')
        self.x_mean = self.x.mean(axis=0)
        x_scale = self.x.std(axis=0)
        self.x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
        self.y_mean = float(self.y.mean())
        y_scale = float(self.y.std())
        self.y_scale = y_scale if y_scale > 1e-12 else 1.0
        self.xs = (self.x - self.x_mean) / self.x_scale
        self.ys = (self.y - self.y_mean) / self.y_scale
        self.design = np.column_stack([np.ones(len(self.ys))])
        if len(self.ys) >= self.xs.shape[1] + 2:
            self.design = np.column_stack([self.design, self.xs])
        self.same_batch = self.groups[:, None] == self.groups[None, :]
        self.params = None
        self.log_marginal_likelihood = None
        self._fit()

    @property
    def n_features(self):
        return self.xs.shape[1]

    def _components(self, params):
        length_scales = np.exp(params[:self.n_features])
        signal_variance = math.exp(params[self.n_features])
        constant = math.exp(params[self.n_features + 1])
        batch_variance = math.exp(params[self.n_features + 2])
        residual_variance = math.exp(params[self.n_features + 3])
        return length_scales, signal_variance, constant, batch_variance, residual_variance

    def _covariance(self, params, jitter=1e-9):
        length_scales, signal_variance, constant, batch_variance, residual_variance = self._components(params)
        k = _kernel(self.xs, self.xs, length_scales, signal_variance, constant)
        k = k + batch_variance * self.same_batch
        k = k + (residual_variance + jitter) * np.eye(len(self.ys))
        return k

    def _profile(self, chol):
        """Generalized least-squares trend coefficients and their covariance."""
        h = self.design
        k_inv_h = cho_solve(chol, h, check_finite=False)
        normal = h.T @ k_inv_h
        normal = normal + np.eye(normal.shape[0]) * 1e-10 * max(np.trace(normal), 1.0)
        normal_chol = cho_factor(normal, lower=True, check_finite=False)
        beta = cho_solve(normal_chol, h.T @ cho_solve(chol, self.ys, check_finite=False), check_finite=False)
        normal_inverse = cho_solve(normal_chol, np.eye(normal.shape[0]), check_finite=False)
        return beta, k_inv_h, normal_inverse

    def _negative_log_marginal_likelihood(self, params):
        try:
            k = self._covariance(params)
            chol = cho_factor(k, lower=True, check_finite=False)
            beta, _, _ = self._profile(chol)
        except (np.linalg.LinAlgError, ValueError):
            return 1e12
        residual = self.ys - self.design @ beta
        alpha = cho_solve(chol, residual, check_finite=False)
        log_det = 2.0 * np.sum(np.log(np.diag(chol[0])))
        return 0.5 * float(residual @ alpha) + 0.5 * log_det + 0.5 * len(self.ys) * math.log(2.0 * math.pi)

    def _fit(self):
        d = self.n_features
        # Bounds are deliberately not wide open.  A synthetic or sparse design can
        # make the marginal likelihood collapse onto a noiseless interpolation,
        # which would report certainty the experiment book cannot justify.  The
        # residual and batch noise variances are floored at a small fraction of
        # the observed response variance (in standardized units), so reported
        # intervals stay honest even when the observations happen to lie on a
        # smooth curve.
        bounds = (
            [(math.log(0.05), math.log(5.0))] * d
            + [(math.log(1e-3), math.log(4.0)),
               (math.log(1e-6), math.log(2.0)),
               (math.log(NOISE_FLOOR), math.log(2.0)),
               (math.log(NOISE_FLOOR), math.log(0.5))]
        )
        starts = []
        # A small deterministic set of starts.  More starts only helped in
        # pathological cases and dominated the runtime of a plan request.
        for length in (0.4, 1.5):
            for batch in (0.15, 0.5):
                starts.append(np.array(
                    [math.log(length)] * d
                    + [math.log(0.8), math.log(0.2), math.log(batch), math.log(0.1)]
                ))
        best = None
        for start in starts:
            result = minimize(
                self._negative_log_marginal_likelihood,
                start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 120, 'ftol': 1e-9, 'gtol': 1e-6},
            )
            value = float(result.fun)
            if best is None or value < best[0] - 1e-9:
                best = (value, np.asarray(result.x))
        if best is None or not math.isfinite(best[0]) or best[0] >= 1e11:
            raise ModelError('响应模型不稳定，无法给出可靠区间')
        self.params = best[1]
        self.log_marginal_likelihood = -best[0]
        k = self._covariance(self.params)
        try:
            self.chol = cho_factor(k, lower=True, check_finite=False)
        except (np.linalg.LinAlgError, ValueError) as error:
            raise ModelError('响应模型协方差无法分解') from error
        self.beta, self.beta_chol, self.trend_covariance = self._profile(self.chol)
        self.alpha = cho_solve(self.chol, self.ys - self.design @ self.beta, check_finite=False)
        length_scales, signal_variance, constant, batch_variance, residual_variance = self._components(self.params)
        self.length_scales = length_scales
        self.signal_variance = signal_variance
        self.constant = constant
        self.batch_variance = batch_variance
        self.residual_variance = residual_variance

    def predict(self, x_new):
        x_new = np.asarray(x_new, dtype=float)
        if x_new.ndim == 1:
            x_new = x_new.reshape(1, -1)
        if x_new.shape[1] != self.n_features:
            raise ModelError('候选组合的输入维度不一致')
        xs_new = (x_new - self.x_mean) / self.x_scale
        design_new = np.column_stack([np.ones(len(xs_new)), xs_new]) if self.design.shape[1] > 1 else np.ones((len(xs_new), 1))
        length_scales, signal_variance, constant, batch_variance, residual_variance = self._components(self.params)
        k = _kernel(self.xs, xs_new, length_scales, signal_variance, constant)
        mean_standard = design_new @ self.beta + k.T @ self.alpha
        solved = cho_solve(self.chol, k, check_finite=False)
        latent_variance = signal_variance + constant - np.sum(k * solved, axis=0)
        trend_term = np.sum((design_new @ self.trend_covariance) * design_new, axis=1)
        latent_variance = np.maximum(latent_variance, 0.0) + np.maximum(trend_term, 0.0)
        predictive_variance = latent_variance + batch_variance + residual_variance
        predictive_variance = np.maximum(predictive_variance, 0.0)
        noise_variance = batch_variance + residual_variance
        information_gain = 0.5 * np.log1p(latent_variance / max(noise_variance, 1e-12))
        return {
            'mean': self.y_mean + self.y_scale * mean_standard,
            'std': self.y_scale * np.sqrt(predictive_variance),
            'knowledge_std': self.y_scale * np.sqrt(latent_variance),
            'batch_shared_std': self.y_scale * math.sqrt(max(batch_variance, 0.0)),
            'information_gain': information_gain,
        }

    def diagnostics(self):
        return {
            'batch_effect_sd': float(self.y_scale * math.sqrt(max(self.batch_variance, 0.0))),
            'residual_sd': float(self.y_scale * math.sqrt(max(self.residual_variance, 0.0))),
            'log_marginal_likelihood': float(self.log_marginal_likelihood),
            'length_scales_standardized': [float(value) for value in self.length_scales],
            'trend_coefficients': [float(value) for value in self.beta],
        }


def fit_pair(x, y_strength, y_loss, groups):
    """Fit one independent model per response.

    No relative weighting between strength and loss is introduced anywhere.
    """
    return {
        'strength': GroupedGP(x, y_strength, groups),
        'loss': GroupedGP(x, y_loss, groups),
    }

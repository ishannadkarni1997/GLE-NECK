from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import BulkModelConfig
from .units import effective_dt_ps


@dataclass(frozen=True)
class JaxStack:
    jax: Any
    jnp: Any
    hk: Any
    optax: Any


def jax_stack_available() -> bool:
    try:
        require_jax_stack()
    except ModuleNotFoundError:
        return False
    return True


def require_jax_stack() -> JaxStack:
    """Import the optional JAX/Haiku/Optax stack only when training is requested."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import haiku as hk
    import jax.numpy as jnp
    import optax

    return JaxStack(jax=jax, jnp=jnp, hk=hk, optax=optax)


def _lag_prior_scores(jnp: Any, l_max: int, dt_ps: float, tau_prior_ps: float):
    tau = jnp.arange(l_max, dtype=jnp.float32) * float(dt_ps)
    if tau_prior_ps <= 0:
        return jnp.zeros_like(tau)
    return -tau / float(tau_prior_ps)


def neural_kernel_model(l_max: int, embed_dim: int = 32, field_scale: float = 1.0, tau_prior_ps: float = 0.0, dt_ps: float = 1.0):
    stack = require_jax_stack()
    jax, jnp, hk = stack.jax, stack.jnp, stack.hk

    def model(e_scalar):
        t = jnp.arange(l_max, dtype=jnp.float32) / l_max
        pos_emb = jnp.stack([jnp.sin(2 * jnp.pi * t), jnp.cos(2 * jnp.pi * t)], axis=1)
        scaled_field = e_scalar / field_scale
        field = scaled_field[None]
        field_emb = hk.Linear(embed_dim)(field)
        field_emb = jnp.broadcast_to(field_emb, (l_max, embed_dim))
        x = jnp.concatenate([field_emb, pos_emb], axis=-1)
        x = hk.nets.MLP([64, 64])(x)
        scores = hk.Linear(1)(x).squeeze(-1)
        scores = scores + _lag_prior_scores(jnp, l_max, dt_ps, tau_prior_ps)
        weights = jax.nn.softmax(scores)
        delta_m = hk.Linear(1)(x).squeeze(-1)
        return delta_m * weights

    return model


def neural_kernel_model_poly(
    l_max: int,
    embed_dim: int = 32,
    poly_degree: int = 3,
    field_scale: float = 1.0,
    tau_prior_ps: float = 0.0,
    dt_ps: float = 1.0,
):
    stack = require_jax_stack()
    jax, jnp, hk = stack.jax, stack.jnp, stack.hk

    def model(e_scalar):
        t = jnp.arange(l_max, dtype=jnp.float32) / (l_max - 1)
        powers = jnp.stack([t**degree for degree in range(1, poly_degree + 1)], axis=1)
        scaled_field = e_scalar / field_scale
        field = scaled_field[None]
        field_emb = hk.Linear(embed_dim)(field)
        field_emb = jnp.broadcast_to(field_emb, (l_max, embed_dim))
        x = jnp.concatenate([field_emb, powers], axis=-1)
        x = hk.nets.MLP([64, 64])(x)
        scores = hk.Linear(1)(x).squeeze(-1)
        scores = scores + _lag_prior_scores(jnp, l_max, dt_ps, tau_prior_ps)
        weights = jax.nn.softmax(scores)
        delta_m = hk.Linear(1)(x).squeeze(-1)
        return delta_m * weights

    return model


def asym_neural_kernel_model_poly(
    l_max: int,
    embed_dim: int = 32,
    poly_degree: int = 3,
    field_scale: float = 1.0,
    tau_prior_ps: float = 0.0,
    dt_ps: float = 1.0,
):
    """Polynomial-time corrective kernel with the zero-field E^2 gate."""
    stack = require_jax_stack()
    jax, jnp, hk = stack.jax, stack.jnp, stack.hk

    def model(e_scalar):
        t = jnp.arange(l_max, dtype=jnp.float32) / (l_max - 1)
        powers = jnp.stack([t**degree for degree in range(1, poly_degree + 1)], axis=1)
        scaled_field = e_scalar / field_scale
        field = scaled_field[None]
        field_emb = hk.Linear(embed_dim)(field)
        field_emb = jnp.broadcast_to(field_emb, (l_max, embed_dim))
        x = jnp.concatenate([field_emb, powers], axis=-1)
        x = hk.nets.MLP([64, 64])(x)
        scores = hk.Linear(1)(x).squeeze(-1)
        scores = scores + _lag_prior_scores(jnp, l_max, dt_ps, tau_prior_ps)
        weights = jax.nn.softmax(scores)
        delta_m = hk.Linear(1)(x).squeeze(-1)
        learned_shape = delta_m * weights
        return (scaled_field**2) * learned_shape

    return model


def e2_direct_neural_kernel_model_poly(
    l_max: int,
    embed_dim: int = 32,
    poly_degree: int = 3,
    field_scale: float = 1.0,
):
    """Direct neural corrective kernel with the smooth zero-field E^2 prefactor."""
    stack = require_jax_stack()
    jnp, hk = stack.jnp, stack.hk

    def model(e_scalar):
        t = jnp.arange(l_max, dtype=jnp.float32) / (l_max - 1)
        powers = jnp.stack([t**degree for degree in range(1, poly_degree + 1)], axis=1)
        scaled_field = e_scalar / field_scale
        u = scaled_field**2
        field = u[None]
        field_emb = hk.Linear(embed_dim)(field)
        field_emb = jnp.broadcast_to(field_emb, (l_max, embed_dim))
        x = jnp.concatenate([field_emb, powers], axis=-1)
        x = hk.nets.MLP([64, 64])(x)
        finite_field_shape = hk.Linear(1)(x).squeeze(-1)
        return u * finite_field_shape

    return model


def e2_lag_embedding_neural_kernel_model(
    l_max: int,
    embed_dim: int = 32,
    field_scale: float = 1.0,
):
    """Direct E^2 neural kernel with learned lag embeddings.

    This removes the low-order polynomial lag bottleneck while preserving the
    smooth zero-field condition through the explicit E^2 prefactor.
    """
    stack = require_jax_stack()
    jax, jnp, hk = stack.jax, stack.jnp, stack.hk

    def model(e_scalar):
        lag_ids = jnp.arange(l_max, dtype=jnp.int32)
        tau = jnp.arange(l_max, dtype=jnp.float32) / (l_max - 1)
        scaled_field = e_scalar / field_scale
        u = scaled_field**2

        lag_emb = hk.Embed(vocab_size=l_max, embed_dim=embed_dim)(lag_ids)
        field_emb = hk.nets.MLP([embed_dim, embed_dim], activation=jax.nn.silu)(u[None])
        field_emb = jnp.broadcast_to(field_emb, (l_max, embed_dim))
        field_value = jnp.broadcast_to(u[None], (l_max, 1))
        x = jnp.concatenate([lag_emb, field_emb, tau[:, None], field_value], axis=-1)
        x = hk.nets.MLP([128, 128, 64], activation=jax.nn.silu)(x)
        finite_field_shape = hk.Linear(1)(x).squeeze(-1)
        return u * finite_field_shape

    return model


def e2_tau_mlp_neural_kernel_model(
    l_max: int,
    embed_dim: int = 32,
    field_scale: float = 1.0,
):
    """Direct E^2 neural kernel with a smooth learned tau embedding."""
    stack = require_jax_stack()
    jax, jnp, hk = stack.jax, stack.jnp, stack.hk

    def model(e_scalar):
        tau = jnp.arange(l_max, dtype=jnp.float32) / (l_max - 1)
        scaled_field = e_scalar / field_scale
        u = scaled_field**2

        tau_emb = hk.nets.MLP([embed_dim, embed_dim], activation=jax.nn.silu)(tau[:, None])
        field_emb = hk.nets.MLP([embed_dim, embed_dim], activation=jax.nn.silu)(u[None])
        field_emb = jnp.broadcast_to(field_emb, (l_max, embed_dim))
        field_value = jnp.broadcast_to(u[None], (l_max, 1))
        x = jnp.concatenate([tau_emb, field_emb, tau[:, None], field_value], axis=-1)
        x = hk.nets.MLP([128, 128, 64], activation=jax.nn.silu)(x)
        finite_field_shape = hk.Linear(1)(x).squeeze(-1)
        return u * finite_field_shape

    return model


def e2_film_tau_neural_kernel_model(
    l_max: int,
    embed_dim: int = 32,
    field_scale: float = 1.0,
):
    """Smooth direct E^2 neural kernel with field-conditioned lag features."""
    stack = require_jax_stack()
    jax, jnp, hk = stack.jax, stack.jnp, stack.hk

    def model(e_scalar):
        tau = jnp.arange(l_max, dtype=jnp.float32) / (l_max - 1)
        scaled_field = e_scalar / field_scale
        u = scaled_field**2

        tau_features = hk.nets.MLP([embed_dim, embed_dim], activation=jax.nn.silu)(tau[:, None])
        field_inputs = jnp.stack([scaled_field, u])[None]
        field_features = hk.nets.MLP([embed_dim, embed_dim], activation=jax.nn.silu)(field_inputs)
        gamma_beta = hk.Linear(2 * embed_dim)(field_features)
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)

        modulated_tau = tau_features * (1.0 + gamma) + beta
        field_value = jnp.broadcast_to(field_inputs, (l_max, 2))
        x = jnp.concatenate([modulated_tau, tau[:, None], field_value], axis=-1)
        x = hk.nets.MLP([128, 64], activation=jax.nn.silu)(x)
        finite_field_shape = hk.Linear(
            1,
            w_init=jnp.zeros,
            b_init=jnp.zeros,
        )(x).squeeze(-1)
        return u * finite_field_shape

    return model


def asym_exponential_basis_kernel_model(
    l_max: int,
    field_scale: float = 1.0,
    basis_taus_ps: tuple[float, ...] = (0.05, 0.1, 0.15, 0.25, 0.4),
    dt_ps: float = 1.0,
):
    """E^2-gated corrective kernel constrained to fixed exponential lag scales."""
    stack = require_jax_stack()
    jax, jnp, hk = stack.jax, stack.jnp, stack.hk
    if not basis_taus_ps:
        raise ValueError("basis_taus_ps must contain at least one positive timescale.")
    if any(tau <= 0 for tau in basis_taus_ps):
        raise ValueError("basis_taus_ps values must be positive.")

    tau = jnp.arange(l_max, dtype=jnp.float32) * float(dt_ps)
    basis_taus = jnp.asarray(basis_taus_ps, dtype=jnp.float32)
    basis = jnp.exp(-tau[:, None] / basis_taus[None, :])
    basis = basis / jnp.maximum(jnp.sum(basis, axis=0, keepdims=True), 1.0e-12)

    def model(e_scalar):
        scaled_field = e_scalar / field_scale
        u = scaled_field**2
        field_inputs = jnp.stack([scaled_field, u])[None]
        hidden = hk.nets.MLP([32, 32], activation=jax.nn.silu)(field_inputs)
        amplitudes = hk.Linear(len(basis_taus_ps))(hidden).squeeze(0)
        return u * (basis @ amplitudes)

    return model


def build_kernel_transform(config: BulkModelConfig = BulkModelConfig(), asymptotic: bool = True):
    """Return a Haiku transformed corrective-kernel model."""
    stack = require_jax_stack()
    dt_ps = effective_dt_ps(config)
    if config.kernel_model == "e2_direct":
        model = e2_direct_neural_kernel_model_poly(
            config.l_max,
            config.embed_dim,
            config.poly_degree,
            config.kernel_field_scale,
        )
    elif config.kernel_model == "asym_exp_basis":
        model = asym_exponential_basis_kernel_model(
            config.l_max,
            config.kernel_field_scale,
            config.kernel_exp_basis_taus_ps,
            dt_ps,
        )
    elif config.kernel_model == "e2_lag_embed":
        model = e2_lag_embedding_neural_kernel_model(
            config.l_max,
            config.embed_dim,
            config.kernel_field_scale,
        )
    elif config.kernel_model == "e2_tau_mlp":
        model = e2_tau_mlp_neural_kernel_model(
            config.l_max,
            config.embed_dim,
            config.kernel_field_scale,
        )
    elif config.kernel_model == "e2_film_tau":
        model = e2_film_tau_neural_kernel_model(
            config.l_max,
            config.embed_dim,
            config.kernel_field_scale,
        )
    elif config.kernel_model == "asym_softmax_poly":
        model = asym_neural_kernel_model_poly(
            config.l_max,
            config.embed_dim,
            config.poly_degree,
            config.kernel_field_scale,
            config.kernel_tau_prior_ps,
            dt_ps,
        )
        if not asymptotic:
            model = neural_kernel_model_poly(
                config.l_max,
                config.embed_dim,
                config.poly_degree,
                config.kernel_field_scale,
                config.kernel_tau_prior_ps,
                dt_ps,
            )
    else:
        raise ValueError(f"Unknown kernel model: {config.kernel_model!r}.")
    return stack.hk.transform(model)


def initialize_kernel_params(config: BulkModelConfig = BulkModelConfig(), asymptotic: bool = True):
    """Initialize corrective-kernel parameters using the notebook-compatible seed."""
    stack = require_jax_stack()
    transformed = build_kernel_transform(config=config, asymptotic=asymptotic)
    key = stack.jax.random.PRNGKey(config.seed)
    params = transformed.init(key, stack.jnp.array(0.0))
    tx = stack.optax.adam(config.learning_rate)
    opt_state = tx.init(params)
    return transformed, params, tx, opt_state

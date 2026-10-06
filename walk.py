"""Random weight motion constrained only by the achieved training loss."""

import jax
import jax.numpy as jnp
import numpy as np


def random_walk(loss, theta0, seed, step_size, steps,
                relative_band=0.02, persistence=0.9, record_every=20):
    reference = loss(theta0)
    tolerance = relative_band * reference
    value_gradient = jax.value_and_grad(loss)

    @jax.jit
    def step(theta, direction, key):
        key, draw = jax.random.split(key)
        noise = jax.random.normal(draw, theta.shape, dtype=theta.dtype)
        noise /= jnp.linalg.norm(noise)
        direction = persistence * direction + jnp.sqrt(1 - persistence**2) * noise
        direction /= jnp.linalg.norm(direction)

        _, normal = value_gradient(theta)
        squared_norm = jnp.vdot(normal, normal)
        # At a stationary point every direction is first-order loss-flat.
        tangent = direction - jnp.vdot(direction, normal) * normal / (squared_norm + 1e-20)
        proposal = theta + step_size * tangent / (jnp.linalg.norm(tangent) + 1e-20)

        def correct(_, point):
            value = loss(point)

            def outside(point):
                value, normal = value_gradient(point)
                # Aim just inside the nearest band edge, never at a behavioral target.
                target = reference + jnp.clip(value - reference, -0.9*tolerance, 0.9*tolerance)
                correction = (value - target) * normal / (jnp.vdot(normal, normal) + 1e-20)
                # Prevent a tiny gradient from producing a large Newton correction.
                correction *= jnp.minimum(1., step_size / (jnp.linalg.norm(correction) + 1e-20))
                return point - correction

            return jax.lax.cond(jnp.abs(value-reference) > tolerance, outside, lambda p: p, point)

        proposal = jax.lax.fori_loop(0, 3, correct, proposal)
        value = loss(proposal)
        accepted = jnp.isfinite(value) & (jnp.abs(value-reference) <= tolerance)
        theta = jnp.where(accepted, proposal, theta)
        return theta, direction, key, accepted

    theta = theta0
    direction = jnp.zeros_like(theta)
    key = jax.random.PRNGKey(seed)
    weights, losses, decisions = [np.array(theta)], [float(reference)], []
    for index in range(steps):
        theta, direction, key, accepted = step(theta, direction, key)
        decisions.append(bool(accepted))
        if (index+1) % record_every == 0 or index+1 == steps:
            weights.append(np.array(theta))
            losses.append(float(loss(theta)))
    return np.array(weights), np.array(losses), np.array(decisions)

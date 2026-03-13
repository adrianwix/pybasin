"""Experiment: auto-derive PARAM_KEYS inside JaxODESystem itself via __init_subclass__.

Goal: no extra base class, no extra inheritance layer. The __init_subclass__ hook
lives directly in JaxODESystem so any subclass written as

    class MyODE(JaxODESystem[MyParams]): ...

gets PARAM_KEYS populated automatically when MyParams is a TypedDict.
"""

from typing import Any, ClassVar, TypedDict, cast, get_args, is_typeddict

import jax.numpy as jnp
from jax import Array

from pybasin.utils import AutoGetStrMixin

# ── Local copy of JaxODESystem with __init_subclass__ built in ───────────────


class JaxODESystem[P](AutoGetStrMixin):
    PARAM_KEYS: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            args = get_args(base)
            if args and is_typeddict(args[0]):
                cls.PARAM_KEYS = tuple(args[0].__annotations__.keys())
                break

    def __init__(self, params: P) -> None:
        self.params = params

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        raise NotImplementedError

    def params_to_array(self) -> Array:
        p_dict = cast(dict[str, Any], self.params)
        return jnp.array([p_dict[k] for k in self.PARAM_KEYS])  # pyright: ignore[reportUnknownMemberType]

    def __call__(self, t: Array, y: Array, args: Any = None) -> Array:
        return self.ode(t, y, args)


# ── Usage: single definition, no PARAM_KEYS needed ───────────────────────────


class DuffingParams(TypedDict):
    delta: float
    k3: float
    A: float


class DuffingJaxODE(JaxODESystem[DuffingParams]):
    def __init__(self, params: DuffingParams) -> None:
        super().__init__(params)

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        delta, k3, amplitude = p[0], p[1], p[2]
        x, x_dot = y[0], y[1]
        return jnp.array([x_dot, -delta * x_dot - k3 * x**3 + amplitude * jnp.cos(t)])


# ── Runtime check ─────────────────────────────────────────────────────────────

assert DuffingJaxODE.PARAM_KEYS == ("delta", "k3", "A"), f"got {DuffingJaxODE.PARAM_KEYS}"

ode = DuffingJaxODE({"delta": 0.08, "k3": 1.0, "A": 0.2})
arr = ode.params_to_array()
assert arr.shape == (3,), f"expected (3,), got {arr.shape}"
print("All assertions passed.")
print(f"DuffingJaxODE.PARAM_KEYS = {DuffingJaxODE.PARAM_KEYS}")


def some_func(a: float, b: float) -> float:
    return a + b

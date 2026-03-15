# pyright: reportUntypedBaseClass=false
import logging
from typing import Any

import numpy as np
import torch
from scipy.integrate import solve_ivp
from sklearn.utils.parallel import Parallel, delayed  # type: ignore[import-untyped]

from pybasin.constants import DEFAULT_CACHE_DIR, UNSET
from pybasin.solvers.base import Solver
from pybasin.solvers.numpy_ode_system import NumpyODESystem

logger = logging.getLogger(__name__)


class ScipyParallelSolver(Solver):
    """
    ODE solver using sklearn's parallel processing with scipy's solve_ivp.

    Uses multiprocessing (loky backend) to solve multiple initial conditions in parallel.
    Each worker solves one trajectory at a time using scipy's solve_ivp.

    Requires a :class:`~pybasin.solvers.numpy_ode_system.NumpyODESystem` subclass. The ODE is passed
    directly to ``solve_ivp`` with no PyTorch-to-NumPy conversion overhead.

    See also: [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
    """

    def __init__(
        self,
        t_span: tuple[float, float] = (0, 1000),
        t_steps: int = 1000,
        device: str | None = None,
        n_jobs: int = -1,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_step: float | None = None,
        cache_dir: str | None = DEFAULT_CACHE_DIR,
        t_eval: tuple[float, float] | None = None,
    ):
        """
        Initialize ScipyParallelSolver.

        :param t_span: Tuple (t_start, t_end) defining the integration interval.
        :param t_steps: Number of evaluation points in the save region.
        :param device: Device to use (only 'cpu' supported).
        :param n_jobs: Number of parallel jobs (-1 for all CPUs).
        :param method: Integration method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', etc).
        :param rtol: Relative tolerance (used by adaptive-step methods only).
        :param atol: Absolute tolerance (used by adaptive-step methods only).
        :param max_step: Maximum step size for the solver.
        :param cache_dir: Directory for caching integration results. ``None`` disables caching.
        :param t_eval: Optional save region ``(save_start, save_end)``. Only time points
            in this range are stored. Must be contained within ``t_span``. If ``None``,
            defaults to ``t_span``.
        """
        if device and "cuda" in device:
            logger.warning(
                "  Warning: ScipyParallelSolver does not support CUDA - falling back to CPU"
            )
            device = "cpu"

        super().__init__(
            t_span,
            t_steps=t_steps,
            device="cpu",
            rtol=rtol,
            atol=atol,
            cache_dir=cache_dir,
            t_eval=t_eval,
        )

        self.n_jobs = n_jobs
        self.method = method
        self.max_step = max_step or (t_span[1] - t_span[0]) / 100

    def _get_cache_config(self) -> dict[str, Any]:
        """Include method and max_step in cache key (rtol/atol handled by base class)."""
        config = super()._get_cache_config()
        config["method"] = self.method
        config["max_step"] = self.max_step
        return config

    def clone(
        self,
        *,
        device: str | None = None,
        t_steps_factor: int = 1,
        cache_dir: str | None | object = UNSET,
    ) -> "ScipyParallelSolver":
        """Create a copy of this solver, optionally overriding device, resolution, or caching.

        Note: ScipyParallelSolver only supports CPU, so the device is always CPU.
        """
        if device and "cuda" in device:
            logger.warning("  Warning: ScipyParallelSolver does not support CUDA - using CPU")
        effective_cache_dir = self._cache_dir if cache_dir is UNSET else cache_dir
        return ScipyParallelSolver(
            t_span=self.t_span,
            t_steps=self.t_steps * t_steps_factor,
            device="cpu",
            n_jobs=self.n_jobs,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
            cache_dir=effective_cache_dir,  # type: ignore[arg-type]
            t_eval=self.t_eval,
        )

    def _integrate(  # type: ignore[override]
        self,
        ode_system: NumpyODESystem[Any],
        y0: torch.Tensor,
        save_ts: torch.Tensor,
        params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate using sklearn parallel processing with scipy's solve_ivp.

        :param ode_system: A NumpyODESystem instance.
        :param y0: Initial conditions with shape ``(batch, n_dims)``.
        :param save_ts: Save-region time points (1D tensor).
        :param params: Pre-expanded parameters of shape ``(B*P, n_params)``.
            Each trajectory uses its own parameter row.
        :return: ``(save_ts, y_values)`` where ``y_values`` has shape ``(t_steps, batch, n_dims)``.
        """
        save_ts_np = save_ts.cpu().numpy()
        y0_np = y0.cpu().numpy()
        batch_size = y0_np.shape[0]
        dtype = y0.dtype

        default_p = ode_system.params_to_array()
        params_np = params.cpu().numpy() if params is not None else None

        def solve_single_trajectory(idx: int) -> np.ndarray:
            """Solve ODE for a single initial condition using scipy's solve_ivp."""
            p = params_np[idx] if params_np is not None else default_p
            # scipy.integrate.solve_ivp has incomplete type stubs
            solution = solve_ivp(  # type: ignore[no-untyped-call]
                fun=ode_system.ode,
                t_span=(self.t_span[0], float(save_ts_np[-1])),
                y0=y0_np[idx],
                method=self.method,  # type: ignore[arg-type]
                t_eval=save_ts_np,
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
                args=(p,),
            )
            return solution.y.T  # type: ignore[no-any-return]

        if batch_size == 1 or self.n_jobs == 1:
            results = [solve_single_trajectory(i) for i in range(batch_size)]
        else:
            # sklearn.utils.parallel has incomplete type stubs
            results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0)(  # type: ignore[misc]
                delayed(solve_single_trajectory)(i)  # type: ignore[misc]
                for i in range(batch_size)
            )

        # Filter out None values and stack
        valid_results = [r for r in results if r is not None]  # type: ignore[misc]
        y_result_np: np.ndarray = np.stack(valid_results, axis=1)  # type: ignore[arg-type]

        y_result = torch.tensor(y_result_np, dtype=dtype, device=self.device)

        return save_ts, y_result

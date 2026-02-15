"""Generic experiment runner with sweep and repetition support.

Uses joblib for parallel execution of independent trials.
"""

from typing import Callable, Dict, List, Any
from itertools import product
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def run_sweep(trial_fn: Callable, param_grid: Dict[str, List[Any]],
              n_reps: int = 500, n_jobs: int = -1,
              desc: str = "Running") -> pd.DataFrame:
    """Run a trial function over a parameter grid with repetitions.

    Parameters
    ----------
    trial_fn : callable
        Function(**params, seed=int) -> dict of results.
    param_grid : dict
        Maps parameter names to lists of values to sweep.
    n_reps : int
        Number of repetitions per parameter setting.
    n_jobs : int
        Number of parallel workers. -1 = all cores, 1 = sequential.
    desc : str
        Progress bar description.

    Returns
    -------
    results : pd.DataFrame
        One row per (parameter_setting, repetition).
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))

    def _run_one(combo, rep):
        params = dict(zip(keys, combo))
        result = trial_fn(**params, seed=rep)
        result["rep"] = rep
        result.update(params)
        return result

    tasks = [(combo, rep) for combo in combos for rep in range(n_reps)]

    if n_jobs == 1:
        results = [_run_one(combo, rep) for combo, rep in tqdm(tasks, desc=desc)]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_one)(combo, rep)
            for combo, rep in tqdm(tasks, desc=desc)
        )

    return pd.DataFrame(results)


def run_trials_parallel(trial_fn: Callable, args_list: list,
                        n_jobs: int = -1, desc: str = "Running") -> list:
    """Run a list of independent trial calls in parallel.

    Parameters
    ----------
    trial_fn : callable
        Function that takes a single argument (a dict or tuple).
    args_list : list
        Each element is passed to trial_fn.
    n_jobs : int
        Parallelism. -1 = all cores.
    desc : str
        Progress bar description.

    Returns
    -------
    results : list of return values from trial_fn.
    """
    if n_jobs == 1:
        return [trial_fn(a) for a in tqdm(args_list, desc=desc)]
    else:
        return Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(trial_fn)(a) for a in tqdm(args_list, desc=desc)
        )

"""Evolutionary batches started from the data: one CMA-ES or DE generation.

A live CMA-ES or DE run keeps its state between generations. Here each block
rebuilds it from the measured points instead: the best ones become the
search distribution (CMA-ES) or the population (DE). With candidates, each
proposed point is replaced by the nearest distinct available row.
"""
from __future__ import annotations

from importlib import metadata

import cma
import numpy as np

from .base import check_inputs, finish, snap


def _ranked(inputs, need, name):
    if len(inputs.y) < need:
        raise ValueError(f'{name} needs at least {need} measured points; start with space_filling')
    return inputs.X[np.argsort(inputs.y, kind='stable')]


def _place(name, inputs, points, settings, info):
    points = np.clip(points, 0., 1.)
    if inputs.candidates is not None:
        return finish(name, inputs, indices=snap(points, inputs.candidates, inputs.open), settings=settings,
                      info=info)
    return finish(name, inputs, X=points, settings=settings, info=info)


def cma_es(X, y, q, *, pending=None, failed=None, candidates=None, dim=None, seed=0, mu=None, sigma=None):
    """One CMA-ES generation of q points, centred on the best measured points.

    mu: how many of the best points shape the start (default max(2, q // 2)),
    weighted by CMA-ES's log-rank weights. The mean is their weighted mean;
    sigma defaults to their root-mean-square spread around it (clipped to
    [0.01, 0.5]); per-coordinate scales follow their spread, so the search
    starts shaped like the best region found. pycma draws the generation
    with its own bound handling in the unit cube.
    """
    inputs = check_inputs(X, y, q, pending, failed, candidates, dim)
    mu = max(2, q // 2) if mu is None else mu
    if isinstance(mu, bool) or not isinstance(mu, (int, np.integer)) or mu < 2:
        raise ValueError(f'mu must be an integer of at least 2, not {mu!r}')
    elite = _ranked(inputs, mu, 'cma_es')[:mu]
    weights = np.log(mu + .5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mean = weights @ elite
    spread = np.sqrt(weights @ (elite - mean) ** 2)
    step = float(np.clip(np.sqrt(np.mean(spread ** 2)), .01, .5)) if sigma is None else float(sigma)
    if not 0 < step <= 1:
        raise ValueError(f'sigma must be in (0, 1], not {sigma!r}')
    stds = np.clip(spread / step, .1, 10.)
    # pycma treats seed 0 as "seed from the clock".
    options = {'popsize': max(q, 2), 'seed': seed + 1, 'bounds': [0., 1.], 'CMA_stds': stds.tolist(),
               'verbose': -9}
    strategy = cma.CMAEvolutionStrategy(mean.tolist(), step, options)
    points = np.array(strategy.ask())[:q]
    settings = {'seed': seed, 'mu': mu, 'sigma': sigma if sigma is not None else 'from the elite spread',
                'library': f'cma {metadata.version("cma")}, popsize {max(q, 2)}, bounds [0, 1]'}
    info = {'mean': mean.tolist(), 'sigma': step, 'stds': stds.tolist()}
    return _place('cma_es', inputs, points, settings, info)


def de(X, y, q, *, pending=None, failed=None, candidates=None, dim=None, seed=0,
       population=None, F=.8, CR=.9, strategy='rand1'):
    """One differential-evolution generation: q trial points from the best measured points.

    population: how many of the best points form the population (default
    max(q, 4)). Each trial takes a target from the population in rank order,
    a mutant (rand1: r1 + F (r2 - r3); best1: best + F (r2 - r3)) and binomial
    crossover with rate CR, keeping at least one mutant coordinate. Trials are
    clipped to the cube.
    """
    inputs = check_inputs(X, y, q, pending, failed, candidates, dim)
    size = max(q, 4) if population is None else population
    if strategy not in ('rand1', 'best1'):
        raise ValueError(f"strategy must be 'rand1' or 'best1', not {strategy!r}")
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 4:
        raise ValueError(f'population must be an integer of at least 4, not {size!r}')
    ranked = _ranked(inputs, 4, 'de')
    pool = ranked[:size]
    rng = np.random.default_rng(seed)
    trials = []
    for i in range(q):
        target = i % len(pool)
        others = rng.choice([j for j in range(len(pool)) if j != target], 3, replace=False)
        base = pool[0] if strategy == 'best1' else pool[others[0]]
        mutant = base + F * (pool[others[1]] - pool[others[2]])
        mask = rng.random(inputs.dim) < CR
        mask[rng.integers(inputs.dim)] = True
        trials.append(np.where(mask, mutant, pool[target]))
    settings = {'seed': seed, 'population': len(pool), 'F': F, 'CR': CR, 'strategy': strategy}
    return _place('de', inputs, np.array(trials), settings, {'population_best': pool[0].tolist()})

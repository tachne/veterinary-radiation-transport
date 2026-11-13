"""One gamma irradiation per session, with views defined by photon number."""
from dataclasses import dataclass
from numbers import Integral
import secrets

import numpy as np

from .model import Photon, Sample, Slab, photon, sample_photons


MAX_TRACKS = 100
MAX_SEED = 2**32 - 1


@dataclass(frozen=True, eq=False)
class Irradiation:
    slab: Slab
    beam: Photon
    sample: Sample
    hvl_passed: np.ndarray | None
    hvl_count: int | None


def current_irradiation(state, slab, energy_kev, count, seed):
    run = state.get("_gamma_irradiation")
    if (run is not None and run.slab == slab and run.beam.energy_kev == energy_kev
            and run.sample.count == count and run.sample.seed == seed):
        return run
    beam = photon(slab, energy_kev)
    sample = sample_photons(slab, beam, count, seed)
    hvl_passed = None
    hvl_count = None
    if beam.hvl <= slab.thickness:
        hvl_passed = sample.distances > beam.hvl
        hvl_passed.setflags(write=False)
        hvl_count = int(np.count_nonzero(hvl_passed))
    # The graph and deferred CSV downloads read the same snapshot.
    sample.distances.setflags(write=False)
    sample.interacted.setflags(write=False)
    run = Irradiation(slab, beam, sample, hvl_passed, hvl_count)
    state["_gamma_irradiation"] = run
    return run


def track_slice(count, first, size):
    """Convert a 1-based start and requested width to a bounded array slice."""
    if not isinstance(first, Integral) or not 1 <= first <= count:
        raise ValueError("開始番号は計算した光子の範囲で指定してください。")
    if not isinstance(size, Integral) or not 1 <= size <= MAX_TRACKS:
        raise ValueError("表示本数は1〜100の範囲で指定してください。")
    return slice(first - 1, min(first - 1 + size, count))


def different_seed(previous):
    # Map a uniform choice from 2**32 - 1 values onto every seed except the current one.
    candidate = secrets.randbelow(MAX_SEED)
    return candidate + int(candidate >= previous)

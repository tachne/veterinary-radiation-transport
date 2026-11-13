"""Straight CSDA path in a homogeneous slab; no low-energy extrapolation."""
import csv
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .beta_spectrum import Emissions, immutable, sample_emissions
from .model import Slab
from .reference import DATA, ElectronTable, electron_table

INITIAL_LOW, PATH_END, STOPPED, RESIDUAL_LOW, NO_PATH = range(5)
STATUS = ("initial_below_table", "residual_in_table", "csda_path_exhausted",
          "residual_below_table", "zero_thickness")
LABELS = ("入射10 keV未満", "残存値を算出", "CSDA経路を消尽", "残存0超〜10 keV未満", "試料厚0")


@lru_cache(maxsize=7)
def beta_table(material):
    base = electron_table(material)
    with (DATA / "beta" / "estar_2p5.csv").open(newline="", encoding="utf-8") as f:
        [extra] = [r for r in csv.DictReader(f) if r["material"] == material]
    columns = (("energy", "energy_MeV"), ("collision", "collision_MeV_cm2_g"),
               ("radiative", "radiative_MeV_cm2_g"), ("total", "total_MeV_cm2_g"),
               ("range_mass", "csda_g_cm2"))
    result = ElectronTable(*(immutable(np.r_[getattr(base, attr), float(extra[key])])
                             for attr, key in columns))
    if (result.energy[0] != .010 or result.energy[-1] != 2.5
            or not np.all(np.diff(result.energy)>0) or not np.all(np.diff(result.range_mass)>0)):
        raise ValueError("ESTARの参照表の範囲が不正です。")
    return result


def _in_table(values, grid, positive_column):
    """Vector log-log interpolation. Exactly preserve knots; out of grid = NaN."""
    x = np.asarray(values, dtype=float)
    ok = (grid[0] <= x) & (x <= grid[-1])
    result = np.full(x.shape, np.nan)
    q = x[ok]
    i = np.minimum(np.searchsorted(grid, q, side="right") - 1, len(grid)-2)
    w = np.log(q/grid[i]) / np.log(grid[i+1]/grid[i])
    result[ok] = np.where(q == grid[i+1], positive_column[i+1],
                          positive_column[i] * (positive_column[i+1]/positive_column[i])**w)
    return result


def energy_after_path(table, density, energy, range_mass, distance_cm):
    """Invert R(E)-rho*s only inside the table. Bounds are not estimates."""
    e, r, s = np.broadcast_arrays(energy, range_mass, distance_cm)
    if np.any(~np.isfinite(s) | (s < 0)):
        raise ValueError("経路長は0以上の有限値で指定してください。")
    supported = np.isfinite(r)
    stop = supported & (s >= r/density)
    end = _in_table(r - density*s, table.range_mass, table.energy)
    # At zero thickness even a subthreshold electron loses exactly no energy.
    end = np.where(s == 0, e, np.where(stop, 0., np.minimum(end, e)))
    state = np.select([s == 0, ~supported, stop, ~np.isfinite(end)],
                      [NO_PATH, INITIAL_LOW, STOPPED, RESIDUAL_LOW], default=PATH_END)
    return end, state.astype(np.uint8)


@dataclass(frozen=True, eq=False)
class BetaRun:
    slab: Slab
    emissions: Emissions
    collision: np.ndarray
    radiative: np.ndarray
    total: np.ndarray
    mass_range: np.ndarray
    path_cm: np.ndarray
    end_energy: np.ndarray
    loss: np.ndarray
    status: np.ndarray


def transport_emissions(slab, emissions):
    table = beta_table(slab.material.key)
    e = emissions.energy
    if np.any((e < 0) | (e > table.energy[-1]) | ~np.isfinite(e)):
        raise ValueError("入射エネルギーがモデルの対象外です。")
    coll, rad, total, r = (_in_table(e, table.energy, values) for values in
                            (table.collision, table.radiative, table.total, table.range_mass))
    end, status = energy_after_path(table, slab.density, e, r, slab.thickness)
    return BetaRun(slab, emissions, *(immutable(a) for a in
                   (coll, rad, total, r, r/slab.density, end, e-end, status)))


def current_beta_run(state, slab, count, seed):
    """Student-local sample; material and thickness only change its transport."""
    old = state.get("_beta_equilibrium_run")
    if old is not None and (old.emissions.count, old.emissions.seed) == (count, seed):
        if old.slab == slab:
            return old
        emissions = old.emissions
    else:
        emissions = sample_emissions(count, seed)
    run = transport_emissions(slab, emissions)
    state["_beta_equilibrium_run"] = run
    return run

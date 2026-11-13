"""Photon first collisions and straight-path electron CSDA. Distances are cm."""
from dataclasses import dataclass
from math import exp, log, sqrt
from numbers import Integral

import numpy as np

from .reference import MATERIALS, Material, bounded, electron_table, log_value, mass_attenuation


@dataclass(frozen=True)
class Slab:
    material: Material
    thickness: float
    density_scale: float = 1.0

    def __post_init__(self):
        if self.material.key not in MATERIALS:
            raise ValueError("未登録の物質です。")
        bounded("厚さ (cm)", self.thickness, 0, 100_000)
        bounded("相対密度", self.density_scale, 0.25, 4.0)

    @property
    def density(self):
        return self.material.density * self.density_scale


@dataclass(frozen=True)
class Photon:
    energy_kev: float
    mass_mu: float
    mu: float
    hvl: float
    transmission: float


def photon(slab, energy_kev):
    mass_mu = mass_attenuation(slab.material.key, energy_kev)
    mu = mass_mu * slab.density
    return Photon(energy_kev, mass_mu, mu, log(2) / mu, exp(-mu * slab.thickness))


@dataclass(frozen=True)
class Sample:
    seed: int
    distances: np.ndarray
    interacted: np.ndarray
    transmitted: int
    fraction: float
    standard_error: float

    @property
    def count(self):
        return len(self.distances)


def sample_photons(slab, beam, count, seed):
    for name, value, lo, hi in (("光子数", count, 1, 200_000),
                                ("乱数シード", seed, 0, 2**32 - 1)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} は整数で指定してください。")
        bounded(name, value, lo, hi)
    rng = np.random.Generator(np.random.PCG64(seed))
    # Sample strictly inside (0,1); extending N preserves its original prefix.
    u = rng.integers(1, 2**53, size=count) / float(2**53)
    depths = -np.log(u) / beam.mu
    interacted = depths <= slab.thickness
    passed = int(count - np.count_nonzero(interacted))
    p = beam.transmission
    return Sample(seed, depths, interacted, passed, passed / count,
                  sqrt(p * (1 - p) / count))


@dataclass(frozen=True)
class Electron:
    energy_kev: float
    collision: float
    radiative: float
    total: float
    range_mass: float
    range_cm: float
    end_kev: float
    stopped: bool

    @property
    def loss_kev(self):
        return self.energy_kev - self.end_kev


def residual_kev(slab, energy_kev, range_mass, distance_cm):
    """Invert remaining CSDA mass range; linear closure to (0,0) below 10 keV."""
    table = electron_table(slab.material.key)
    x = np.asarray(distance_cm, dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("経路長には0以上の有限値が必要です。")
    remaining = np.maximum(range_mass - slab.density * x, 0.0)
    low_end = table.range_mass[0]
    log_r = np.log(np.maximum(remaining, low_end))
    energy = 1000 * np.exp(np.interp(log_r, np.log(table.range_mass), np.log(table.energy)))
    energy = np.where(remaining < low_end, 10 * remaining / low_end, energy)
    energy = np.where(x >= range_mass / slab.density, 0.0, energy)
    return np.where(x == 0, energy_kev, np.clip(energy, 0, energy_kev))


def electron(slab, energy_kev):
    bounded("電子の初期エネルギー (keV)", energy_kev, 100, 2000)
    table = electron_table(slab.material.key)
    collision, radiative, total, rng = (
        log_value(energy_kev / 1000, table.energy, field)
        for field in (table.collision, table.radiative, table.total, table.range_mass))
    length = rng / slab.density
    residual = float(residual_kev(slab, energy_kev, rng, slab.thickness))
    return Electron(energy_kev, collision, radiative, total, rng, length, residual,
                    slab.thickness >= length)

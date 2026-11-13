"""Conditional beta spectra per emitted electron, from ICRP 107 via MIRDspecs."""
import csv
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import numpy as np

from .reference import DATA

NUCLIDES = ("Sr-90", "Y-90")
MAX_COUNT = 100_000
MAX_SEED = 2**32 - 1
LAB_VERSION = "0.2.1"
BETA_MODEL_VERSION = "SrY-CSDA-0.2.0"
EQUILIBRIUM = "A(Sr-90) = A(Y-90); P(Sr-90) = P(Y-90) = 0.5"
MIRD = "https://mirdsoft.org/products/MIRDspecs/MIRDspecs_HTMLs/{}.htm"


def immutable(a):
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class Spectrum:
    isotope: str
    energy: np.ndarray  # MeV
    density: np.ndarray  # conditional PDF in 1/MeV
    cdf: np.ndarray  # probabilities at the knots
    raw_area: float
    mean: float
    reference_mean: float  # independent, yield-weighted beta branches

    def pdf(self, energy):
        return np.interp(energy, self.energy, self.density, left=0., right=0.)

    def quantile(self, probability):
        """Invert the integral of the *linear PDF*, not a linearized CDF."""
        u = np.asarray(probability, dtype=float)
        if np.any(~np.isfinite(u) | (u < 0.) | (u > 1.)):
            raise ValueError("累積確率は0〜1の有限値で指定してください。")
        i = np.clip(np.searchsorted(self.cdf, u, side="right") - 1, 0, len(self.cdf)-2)
        dx = self.energy[i+1] - self.energy[i]
        y0, y1 = self.density[i], self.density[i+1]
        # Normalize within this interval. Rationalizing the quadratic also
        # handles a constant density, and avoids cancellation near a flat slope.
        f = (u - self.cdf[i]) / (self.cdf[i+1] - self.cdf[i])
        yf = np.sqrt(np.maximum((1-f)*y0**2 + f*y1**2, 0.))
        offset = np.divide(f*(y0+y1), y0+yf, out=np.zeros_like(u), where=(y0+yf)>0)
        return np.where(u == 1., self.energy[-1], self.energy[i] + dx*offset)


@lru_cache(maxsize=2)
def spectrum(isotope):
    if isotope not in NUCLIDES:
        raise ValueError("対象核種はSr-90とY-90です。")
    path = DATA / "beta"
    with (path / f"{isotope}_source.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    x = np.array([float(r["Energy [MeV]"]) for r in rows])
    raw = np.array([float(r["Probability density [#/MeV/nt]"]) for r in rows])
    dx = np.diff(x)
    if (not np.isfinite(x).all() or not np.isfinite(raw).all() or x[0] != 0.
            or np.any(dx <= 0.) or np.any(raw < 0.) or raw[-1] != 0.):
        raise ValueError(f"{isotope}: スペクトル表を確認してください。")
    areas = dx * (raw[:-1] + raw[1:]) / 2
    if np.any(areas <= 0.):
        raise ValueError("確率質量のない区間があります。")
    cumulative = np.r_[0., np.cumsum(areas)]
    area = float(cumulative[-1])
    # Exact first moment of each piecewise-linear interval.
    moment = np.sum(x[:-1]*areas + dx**2*(raw[:-1]+2*raw[1:])/6)
    with (path / f"{isotope}_summary.csv").open(newline="", encoding="utf-8-sig") as f:
        branches = [(float(r["Energy [MeV]"]), float(r["Yield [#/nt]"]))
                    for r in csv.DictReader(f) if r["Type (detailed)"] == "Beta"]
    ref = sum(e*w for e, w in branches) / sum(w for _, w in branches)
    return Spectrum(isotope, immutable(x), immutable(raw/area), immutable(cumulative/area),
                    area, float(moment/area), ref)


@dataclass(frozen=True, eq=False)
class Emissions:
    seed: int
    isotope: np.ndarray  # 0 and 1 index NUCLIDES
    energy: np.ndarray   # all draws, including those below ESTAR's lower limit

    @property
    def count(self):
        return len(self.energy)


def sample_emissions(count, seed):
    for value, lo, hi in ((count, 1, MAX_COUNT), (seed, 0, MAX_SEED)):
        if isinstance(value, bool) or not isinstance(value, Integral) or not lo <= value <= hi:
            raise ValueError("粒子数または乱数シードが整数の許容範囲外です。")
    # One independent pair per particle; increasing N preserves existing IDs.
    u = np.random.Generator(np.random.PCG64(seed)).random((count, 2))
    ids = (u[:, 0] >= .5).astype(np.uint8)
    e = np.empty(count)
    for j, name in enumerate(NUCLIDES):
        mask = ids == j
        e[mask] = spectrum(name).quantile(u[mask, 1])
    return Emissions(int(seed), immutable(ids), immutable(e))

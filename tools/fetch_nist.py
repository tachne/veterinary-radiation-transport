"""Rebuild the small offline teaching tables using the public NIST forms."""
import argparse
import csv
import html
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

HOST = "https://physics.nist.gov"
XCOM = HOST + "/cgi-bin/Xcom/xcom3_3"
ESTAR = HOST + "/cgi-bin/Star/e_table-t.pl"
CUSTOM = HOST + "/cgi-bin/Star/e_table-ut.pl"
WEBBOOK = "https://webbook.nist.gov/cgi/cbook.cgi?ID=C122327"
EDGE_SOURCE = HOST + "/PhysRefData/XrayMassCoef/ElemTab/z82.html"
# The XCOM HTML displays both sides to four significant digits.
PB_EDGE_MEV = 0.0880045
MATERIALS = (
    ("air", "空気", "乾燥空気（海面付近）", "104", 0.0012),
    ("oil", "油", "トリオレイン C₅₇H₁₀₄O₆", None, 0.92),
    ("water", "水", "液体水", "276", 1.00),
    ("bone", "皮質骨", "ICRP 皮質骨", "120", 1.85),
    ("al", "アルミニウム", "元素 Al", "013", 2.70),
    ("fe", "鉄", "元素 Fe", "026", 7.87),
    ("pb", "鉛", "元素 Pb", "082", 11.34),
)
SYMBOLS = {1:"H", 6:"C", 7:"N", 8:"O", 12:"Mg", 13:"Al", 15:"P", 16:"S",
           18:"Ar", 20:"Ca", 26:"Fe", 30:"Zn", 82:"Pb"}
# Additional nodes to improve interpolation between standard XCOM energies.
EXTRA_KEV = (32, 35, 38, 45, 55, 65, 70, 75, 85, 90, 95, 110, 120, 130, 140,
             175, 225, 250, 275, 350, 450, 550, 650, 700, 750, 850, 900, 950,
             1100, 1200, 1350, 1450, 1600, 1750, 1900)
CHECK_KEV = (37, 73, 87.99, 88.02, 320, 662, 1370)


def request(url, fields=None):
    cmd = ["curl", "--fail", "--silent", "--show-error", "--max-time", "35",
           "--retry", "2"]
    if fields is not None:
        cmd += ["--data", urlencode(fields)]
    return subprocess.check_output(cmd + [url]).decode("latin-1")


def parse_xcom(page):
    rows = []
    for tr in re.findall(r"<tr\b[^>]*>(.*?)</tr>", page, re.I | re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", x)).strip()
                 for x in re.findall(r"<td\b[^>]*>(.*?)</td>", tr, re.I | re.S)]
        if len(cells) != 9 or not re.fullmatch(r"\d\.\d+E[+-]\d+", cells[1]):
            continue
        energy, value = float(cells[1]), float(cells[7])
        side = ""
        if cells[0]:
            if not (cells[0].split() == ["82", "K"] and rows[-1][0] == energy):
                raise ValueError(f"Unexpected absorption edge: {cells[:2]}")
            rows[-1] = (PB_EDGE_MEV, "below", rows[-1][2])
            energy, side = PB_EDGE_MEV, "above"
        rows.append((energy, side, value))
    return rows


def parse_estar(page):
    pre = re.search(r"<pre>(.*?)</pre>", page, re.I | re.S)
    if pre is None:
        raise ValueError("ESTAR returned no table: " + re.sub(r"<[^>]+>", " ", page)[-500:])
    rows = []
    for line in re.split(r"<br\s*/?>|\n", pre[1], flags=re.I):
        numbers = line.strip().split()
        if len(numbers) == 7 and all(re.fullmatch(r"\d\.\d+E[+-]\d+", n) for n in numbers):
            e, col, rad, total, rng = map(float, numbers[:5])
            if 0.01 <= e <= 2:
                rows.append((e, col, rad, total, rng))
    if not rows or (rows[0][0], rows[-1][0]) != (0.01, 2.0):
        raise ValueError("Incomplete ESTAR table")
    return rows


def write_table(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def collect(out, raw):
    out.mkdir(exist_ok=True, parents=True)
    if raw:
        raw.mkdir(exist_ok=True, parents=True)
    def get(name, url, fields=None):
        page = request(url, fields)
        if raw:
            (raw / (name + ".html")).write_bytes(page.encode("latin-1"))
        return page

    metadata, photons, electrons, checks = [], [], [], []
    for key, name, description, matno, density in MATERIALS:
        if matno:
            source = HOST + "/cgi-bin/Star/compos.pl?matno=" + matno
            comp = get(key + "_comp", source)
            fractions = re.findall(r"<td>(\d+)</td>\s*<td>([\d.]+)</td>", comp)
            if not fractions:
                raise ValueError("No composition for " + key)
            formula = "\r\n".join(f"{SYMBOLS[int(z)]} {w}" for z, w in fractions)
            props = re.findall(r"<td align=left>([\d.+E-]+)</td>", comp)
            ref_density, excitation = map(float, props[:2])
            epage = get(key + "_estar", ESTAR,
                        {"matno":matno, "ShowDefault":"on", "Energies":""})
        else:
            source, formula, ref_density = WEBBOOK, "C57H104O6 1", density
            base = {"Name":"Triolein", "Density":str(density), "Formulae":formula}
            form = get("oil_definition", HOST + "/cgi-bin/Star/estar-ut.pl", base)
            excitation = float(re.search(r'name="I" value="([\d.]+)"', form)[1])
            epage = get("oil_estar", CUSTOM, {"Name":"Triolein", "Density":str(density),
                        "pairnum":"0", "line0":formula, "I":str(excitation),
                        "ShowDefault":"on", "Energies":""})
        erows = parse_estar(epage)
        electrons.extend((key, *row) for row in erows)
        inputs = {"Name":key, "Formulae":formula, "Graph0":"on", "NumAdd":"1",
                  "WindowXmin":"0.03", "WindowXmax":"2", "ResizeFlag":"on"}
        def xpage(suffix, grid, standard=False):
            fields = {**inputs, "Energies":"\r\n".join(f"{e/1000:.6g}" for e in grid)}
            if standard:
                fields["Output"] = "on"
            return get(key + suffix, XCOM, fields)
        page = xpage("_xcom", EXTRA_KEV, True)
        xrows = parse_xcom(page)
        if not xrows or xrows[0][0] != 0.03 or xrows[-1][0] != 2:
            raise ValueError("Incomplete XCOM data for " + key)
        photons.extend((key, *row) for row in xrows)
        crows = parse_xcom(xpage("_holdout", CHECK_KEV))
        if len(crows) != len(CHECK_KEV):
            raise ValueError(f"Incomplete independent XCOM check for {key}: {crows}")
        checks.extend((key, e, v) for e, _, v in crows)
        weights = {SYMBOLS[int(z)]:float(w) for z,w in
                   re.findall(r"Z=\s*(\d+)\s*:\s*([\d.]+)", page)}
        if abs(sum(weights.values()) - 1) > 0.00001:
            raise ValueError("Incomplete XCOM composition: " + key)
        metadata.append(dict(key=key, name=name, description=description, density=density,
                             reference_density=ref_density, excitation_eV=excitation,
                             composition=weights, composition_url=source, estar_id=matno))
        print(key, len(xrows), len(erows), "I =", excitation, flush=True)

    write_table(out/"xcom.csv", ("material","energy_MeV","edge_side","mass_mu_cm2_g"), photons)
    write_table(out/"estar.csv", ("material","energy_MeV","collision_MeV_cm2_g",
                "radiative_MeV_cm2_g","total_MeV_cm2_g","csda_g_cm2"), electrons)
    write_table(out/"xcom_check.csv", ("material","energy_MeV","mass_mu_cm2_g"), checks)
    (out/"materials.json").write_text(json.dumps({
        "retrieved":datetime.now(ZoneInfo("Asia/Tokyo")).date().isoformat(),
        "xcom":XCOM, "estar":ESTAR, "estar_custom":CUSTOM,
        "lead_edge_energy_source":EDGE_SOURCE, "materials":metadata},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True, help="destination for the new tables")
    cli.add_argument("--raw", type=Path, help="optional archive of the original NIST responses")
    args = cli.parse_args()
    collect(args.out, args.raw)

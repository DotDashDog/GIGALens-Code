#!/usr/bin/env python
"""Turn the hand-edited `improved_sersic_carousel.json` into real JSON.

Five separate problems in the file as saved, fixed here rather than by hand so the
fix is auditable and rerunnable:

1. It is a Python dict *repr*, not JSON -- single quotes throughout.
2. `Array(3.086, dtype=float64, weak_type=True)` reprs, from a `to_params` dump that
   was printed rather than serialised.
3. A genuine syntax error: the `source3` plane is missing the two closing braces that
   end its `light` dict and the plane itself, so `source4_5` .. `source11` were all
   nested INSIDE `source3['light']`. It does not parse at all.
4. `source6` and `source9` are swapped: the plane at z = 1.506 is labelled `source6`
   and the one at z = 1.656 is labelled `source9`, but the canonical mapping
   (`translate_old_params.STRUCTURE["source_ids"]`) is 1.506 -> source 9 and
   1.656 -> source 6. The *values* are right where they are -- the z = 1.506 block's
   centre (-10.16, -15.58) sits under source9's prior, Normal(-8.9, 0.3) /
   Normal(-13.8, 0.3), and the z = 1.656 block's (2.70, 3.44) under source6's
   Normal(2.8, 0.3) / Normal(3.1, 0.3) -- so only the two labels move, never a value.
   Left alone this silently pairs each of the two with the other's real cutout, PSF
   and noise model, and every panel still renders.
5. `cosmo` has no `wa`, but the model's cosmology is `w0waCDM_Cosmo` (H0, Om0, k, w0,
   wa). Filled with 0.0, the LCDM value implied by the file's own w0 = -1.

Run: python clean_json.py IN.json OUT.json
"""
from __future__ import annotations

import ast
import json
import sys
from collections import OrderedDict

# The two plane labels that are swapped, and the light Component inside each (both
# planes hold exactly one Sersic, named after the plane).
SWAP = {"source6": "source9", "source9": "source6"}

# Canonical plane redshift -> source ID, copied from translate_old_params.STRUCTURE so
# this script stays runnable without importing gigalens. Cross-checked below.
SOURCE_IDS = {
    0.962: "1_2", 1.166: "3", 1.432: "4_5", 1.506: "9", 1.627: "7",
    1.656: "6", 3.086: "12_13", 3.549: "8", 4.090: "11",
}


def _repair_source3(text: str) -> str:
    """Insert the two closing braces the `source3` plane is missing (problem 3)."""
    needle = "    'Ie':0.13},\n  'source4_5':"
    if needle not in text:
        raise SystemExit(
            "source3 repair: expected the unbalanced `'Ie':0.13},` / `'source4_5':` "
            "boundary and did not find it. The file changed; re-derive the fix rather "
            "than trusting this script.")
    return text.replace(needle, "    'Ie': 0.13}}},\n  'source4_5':", 1)


def _strip_array_reprs(text: str) -> str:
    """`Array(1.23, dtype=float64, ...)` -> `1.23` (problem 2).

    Done on the AST, not with a regex, so a malformed call raises instead of being
    silently half-rewritten.
    """
    tree = ast.parse(text, mode="eval")

    class Unwrap(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "Array":
                if not node.args:
                    raise SystemExit("Array(...) repr with no positional value")
                return node.args[0]
            raise SystemExit(f"unexpected call {ast.dump(node.func)} in the params dump")

    return Unwrap().visit(tree)


def _check_names(params: dict) -> None:
    """Every plane's name must be the one its redshift implies, and each source plane's
    light Components must be named after it (problem 4's guard, kept after the fix so a
    future edit cannot reintroduce the swap)."""
    for name, plane in params["planes"].items():
        z = plane["geometry"]["redshift"]
        if name == "cluster":
            continue
        sid = next((s for zz, s in SOURCE_IDS.items() if abs(zz - z) < 5e-4), None)
        if sid is None:
            raise SystemExit(f"plane {name!r}: redshift {z:g} is not a known source redshift")
        if name != f"source{sid}":
            raise SystemExit(
                f"plane {name!r} sits at z = {z:g}, which is source {sid} -- expected "
                f"plane name 'source{sid}'. Name/redshift swap, like the source6/source9 "
                "one this script exists to fix.")
        expected = {f"source{n}" for n in sid.split("_")}
        got = set(plane["light"])
        if got != expected:
            raise SystemExit(
                f"plane {name!r}: light Components {sorted(got)} != {sorted(expected)}")


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src).read()

    tree = _strip_array_reprs(_repair_source3(text))
    params = ast.literal_eval(tree)

    # -- problem 4: swap the two mislabelled planes, values untouched ----------------
    planes = params["planes"]
    for old, new in SWAP.items():
        if old not in planes:
            raise SystemExit(f"expected a plane named {old!r} to relabel")
    relabelled = {}
    for name, plane in planes.items():
        new_name = SWAP.get(name, name)
        if new_name != name:
            light = plane["light"]
            if set(light) != {name}:
                raise SystemExit(
                    f"plane {name!r} holds light {sorted(light)}, not the single "
                    f"Component {name!r} the relabel assumes")
            plane = {**plane, "light": {new_name: light[name]}}
        relabelled[new_name] = plane

    # Planes in ascending redshift -- the order the new API assembles them in
    # (observer -> source) and the order the per-plane cutouts must be built in.
    params["planes"] = OrderedDict(
        sorted(relabelled.items(), key=lambda kv: kv[1]["geometry"]["redshift"]))

    # -- problem 5: w0waCDM needs wa -------------------------------------------------
    params["cosmo"].setdefault("wa", 0.0)
    params["cosmo"] = {k: float(params["cosmo"][k])
                       for k in ("H0", "Om0", "k", "w0", "wa")}

    _check_names(params)

    with open(dst, "w") as f:
        json.dump(params, f, indent=2)
        f.write("\n")
    print(f"wrote {dst}")
    for name, plane in params["planes"].items():
        kind = "mass" if "mass" in plane else "light"
        print(f"  {name:<12} z = {plane['geometry']['redshift']:<6.3f} "
              f"{kind}: {', '.join(plane[kind])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

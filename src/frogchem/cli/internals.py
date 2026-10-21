#!/usr/bin/env python3
"""Measure distances, angles and dihedrals from geometries.

Each -i set of 2, 3 or 4 atom indices (0-based) defines a distance (Å), an angle or a
dihedral (degrees). Every conformer of every input file is measured; indices must be
consistent across all files. With -c, the coordinates are Boltzmann-averaged (circular
mean for dihedrals, linear and r⁻⁶ NOE average for distances).
"""

from collections import defaultdict
from typing import List

import numpy as np

from frogchem.cli.base import CLICommand, Guide, Output, ENSEMBLE_WEIGHTS
from frogchem.exceptions import InputError
from frogchem.filetypes import load_structures
from frogchem.math_geometry import calculate_distance, calculate_angle, calculate_dihedral
from frogchem.printer import print_info, print_warning


# Map number of atoms to function, label and unit
INTERNAL_FUNCS = {
    2: (calculate_distance, 'B', 'Å'),  # Bond / distance
    3: (calculate_angle, 'A', '°'),     # Angle
    4: (calculate_dihedral, 'D', '°'),  # Dihedral
}
_TYPE_NAMES = {'B': 'Distance', 'A': 'Angle', 'D': 'Dihedral'}


def get_label_str(atoms_symbols: List[str], idxs: List[int]) -> str:
    """Label of an internal coordinate with its unit, e.g. 'B C0-O1 (Å)'."""
    _, type_label, unit = INTERNAL_FUNCS[len(idxs)]
    atom_chain = "-".join(f"{atoms_symbols[idx]}{idx}" for idx in idxs)
    return f'{type_label} {atom_chain} ({unit})'


class Internals(CLICommand, name='frog-internals'):
    """Measure distances, angles and dihedrals from geometries.

Each -i set of 2, 3 or 4 atom indices (0-based) defines a distance (Å), an angle or a
dihedral (degrees). Every conformer of every input file is measured; indices must be
consistent across all files. With -c, the coordinates are Boltzmann-averaged (circular
mean for dihedrals, linear and r⁻⁶ NOE average for distances).
"""
    category = 'geometry'
    supports_ensemble = True
    _auto_ensemble_average = False
    outputs = [
        Output('table', 'Internal Coordinates',
               'One row per structure/conformer, one column per coordinate (Å or degrees); '
               'with -c only if --individual is given'),
        ENSEMBLE_WEIGHTS,
        Output('table', 'Ensemble-Averaged Internal Coordinates',
               'One row per coordinate: Boltzmann-averaged value and r⁻⁶ NOE distance (Å)', when='-c'),
    ]
    guide = Guide(
        why="""
            Measure the geometric parameters you discuss in a paper — bond lengths, angles, dihedrals
            (torsion angles) — for one structure or for every conformer of an ensemble at once, instead of
            clicking in a viewer. Typical uses: the M–L distance in all conformers, a forming bond in a
            transition state, the dihedral that distinguishes two rotamers, or a Boltzmann-averaged
            distance to compare with an NOE (nuclear Overhauser effect) restraint from NMR.
            """,
        needs="""
            * One or more geometry files: `.xyz` (multi-structure allowed), ORCA or Gaussian outputs
              (last geometry). All files must have the **same atom order**, because the same indices are
              used for every file.
            * The atom numbers, **0-based** (first atom = 0): 2 atoms for a distance, 3 for an angle, 4 for
              a dihedral.
            * For averages (`-c`): a checkpoint with Boltzmann populations — `checkpoint.json` from the
              conformer-ensemble workflow, or a flat JSON like `{"1": 45.2, "2": 30.1}` (conformer number →
              population %) or a CSV. See *Conformer & Ensemble Analysis* in the user guide.
            """,
        how=[
            "Pick the atoms in a viewer and write one `-i` per coordinate, e.g. "
            "`frog-internals ensemble.xyz -i 0 50 -i 50 0 83 -i 50 0 4 3` (distance 0–50, angle 50–0–83, "
            "dihedral 50–0–4–3).",
            "Read the table: one row per structure (`<file> - Conf <n>`, n 0-based), one column per "
            "coordinate; the column header repeats the atoms with their element, e.g. `B Ag0-P50 (Å)`, so you "
            "can check you picked the right ones.",
            "For a spreadsheet use `--format csv` (or `--table-file values.csv` to keep the screen output too).",
            "For an ensemble average add `-c checkpoint.json`: the weights table and the averaged values are "
            "printed; add `--individual` to also see the per-conformer table.",
        ],
        expect="""
            Distances in Å (`B`), angles and dihedrals in degrees (`A`, `D`). Dihedrals run from −180° to
            +180°: the sign tells the sense of rotation, so +60° and −60° are mirror-related (gauche+ /
            gauche−), and +179° and −179° are almost the same geometry.
            Example (`ensemble.xyz`, conformer 0): `B Ag0-P50` 2.619 Å, `A P50-Ag0-P83` 89.575°,
            `D P50-Ag0-C4-N3` −72.307°.

            With `-c`, structures are matched to the checkpoint by their `conf_<N>` folder or, for a multi-XYZ
            file, by position (structure 0 ↔ conformer 1, structure 1 ↔ conformer 2, ...). **Check the
            *Ensemble Weights* table**: a conformer with 0 % weight was not found in the checkpoint. With
            populations 40/25/15/10/10 % on the first five conformers, the example gave
            ⟨Ag0–P50⟩ = 2.623 Å and ⟨P50–Ag0–C4–N3⟩ = −41.3°.

            * Dihedrals are averaged as angles (circular mean), so +170° and −170° average to 180°, not 0°.
            * For distances a second value, ⟨r⁻⁶⟩⁻¹ᐟ⁶, is given: this is how NOE intensities average and it
              is always ≤ the plain average (short distances dominate). Compare it with NOE-derived distances.
            * `Atom indices ... out of range` means 1-based numbers or a file with fewer atoms.
            """,
    )
    examples = [
        ('frog-internals opt.xyz -i 0 1 -i 0 1 2', 'Distance 0-1 and angle 0-1-2 for every conformer'),
        ('frog-internals ensemble.xyz -i 0 1 2 3 --format csv', 'Dihedral 0-1-2-3 of each conformer as CSV'),
        ('frog-internals conf_*/protocol_3/opt.xyz -i 4 7 -c checkpoint.json',
         'Boltzmann-averaged distance and NOE distance between atoms 4 and 7'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help='XYZ, ORCA or Gaussian file(s)')
        parser.add_argument(
            '-i', '--indices', nargs='+', type=int, required=True, action='append', metavar='IDX',
            help='Atom indices, 0-based: 2 (distance), 3 (angle) or 4 (dihedral); repeatable',
        )

    def run(self, args):
        for idxs in args.indices:
            if len(idxs) not in INTERNAL_FUNCS:
                raise InputError(f"Invalid index set {idxs}: need 2, 3 or 4 atom indices")

        files = self.require_files(args.file)
        # internal_calculated[structure label][parameter label] = value
        internal_calculated = defaultdict(dict)

        for p in files:
            file = str(p)
            print_info(f"Reading {file}...")
            structs = load_structures(file)
            if not structs:
                print_warning(f"No geometry found in {file}")
                continue

            ref_symbols = structs[0][0]
            for idxs in args.indices:
                if max(idxs) >= len(ref_symbols) or min(idxs) < 0:
                    raise InputError(f"Atom indices {idxs} out of range for {file} ({len(ref_symbols)} atoms)")

            for i, (_, conf_geom) in enumerate(structs):
                row_label = f'{file} - Conf {i}'
                for idxs in args.indices:
                    calc_func = INTERNAL_FUNCS[len(idxs)][0]
                    val = calc_func(*[conf_geom[idx] for idx in idxs])
                    internal_calculated[row_label][get_label_str(ref_symbols, idxs)] = float(val)

        all_params = sorted({k for sub in internal_calculated.values() for k in sub})
        if not all_params:
            print_warning("No parameters calculated")
            return

        table_rows = [[label] + [data.get(param) for param in all_params]
                      for label, data in internal_calculated.items()]
        headers = ["Structure"] + all_params

        if args.checkpoint:
            self._run_ensemble_average(internal_calculated, all_params)
            if not args.individual:
                return

        self.output(table_rows, headers=headers, title="Internal Coordinates")

    def _run_ensemble_average(self, internal_calculated, all_params):
        print_info(f"Loading Boltzmann populations from checkpoint: {self.args.checkpoint}")
        labels = list(internal_calculated.keys())
        ens = self.build_ensemble(labels)

        self.output_weights(ens)
        weights = ens.weights

        avg_headers = ["Parameter", "Type", "⟨Value⟩", "NOE ⟨r⁻⁶⟩⁻¹ᐟ⁶ (Å)"]
        avg_rows = []

        for param in all_params:
            ptype = param[0]
            vals = [internal_calculated[label].get(param, np.nan) for label in labels]
            valid = [(w, v) for w, v in zip(weights, vals) if v is not None and not np.isnan(v)]
            if not valid:
                continue

            v_weights, v_vals = zip(*valid)
            sum_w = sum(v_weights)
            if sum_w <= 0:
                continue
            norm_w = [w / sum_w for w in v_weights]

            noe = None
            if ptype == 'D':
                # Circular mean for dihedral angles
                rads = np.radians(v_vals)
                s_sum = sum(w * np.sin(r) for w, r in zip(norm_w, rads))
                c_sum = sum(w * np.cos(r) for w, r in zip(norm_w, rads))
                avg = float(np.degrees(np.arctan2(s_sum, c_sum)))
            else:
                avg = float(sum(w * v for w, v in zip(norm_w, v_vals)))
                if ptype == 'B':
                    inv6_sum = sum(w * (v ** -6) for w, v in zip(norm_w, v_vals) if v > 0)
                    noe = float(inv6_sum ** (-1.0 / 6.0)) if inv6_sum > 0 else None
            avg_rows.append([param, _TYPE_NAMES.get(ptype, 'Coordinate'), avg, noe])

        self.output(avg_rows, headers=avg_headers,
                    title="Ensemble-Averaged Internal Coordinates (Boltzmann Weighted)")


def parse_args():
    return Internals().build_parser().parse_args()

def main():
    Internals().main()


if __name__ == '__main__':
    main()

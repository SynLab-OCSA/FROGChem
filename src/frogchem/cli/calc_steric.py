"""Compute the buried volume (%V_bur) around a center and draw steric maps.

The geometry is oriented with the center at the origin, the +Z axis towards the centroid
of the -z atoms and the X axis towards the -x atoms. %V_bur is computed with the SambVca
model (Bondi radii scaled by 1.17) in a sphere around the center, split into quadrants
and octants. With -m, a StericRender SVG steric map is written for each input.
"""

import logging

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError
from frogchem.filetypes import load_calculation
from frogchem.printer import print_info, print_error, print_warning
from frogchem.steric_maps import (
    get_sambvca_radii,
    orient_positions,
    compute_buried_volume,
    compute_steric_map,
    generate_svg_overlay,
)

logger = logging.getLogger(__name__)

QUADRANTS = ("NE", "NW", "SW", "SE")
OCTANTS = tuple(f"{side} {q}" for side in ("Front", "Back") for q in QUADRANTS)


class CalcSteric(CLICommand, name='frog-steric'):
    """Compute the buried volume (%V_bur) around a center and draw steric maps.

The geometry is oriented with the center at the origin, the +Z axis towards the centroid
of the -z atoms and the X axis towards the -x atoms. %V_bur is computed with the SambVca
model (Bondi radii scaled by 1.17) in a sphere around the center, split into quadrants
and octants. With -m, a StericRender SVG steric map is written for each input.
"""
    category = 'geometry'
    outputs = [
        Output('table', 'Steric Descriptors (SambVca Model)',
               'One row per file: center, sphere radius (Å), total and per-quadrant %V_bur'),
        Output('table', 'Octant Buried Volume',
               'One row per file: %V_bur of the front (+Z) and back (-Z) octants'),
        Output('file', '<stem>_steric_map.svg', 'StericRender steric map overlaid on the molecule', when='-m'),
        Output('file', '<stem>_steric_map.png', 'PNG rendering of the steric map (needs cairo)', when='-m --png'),
    ]
    guide = Guide(
        why="""
            Quantify how crowded the space around a metal (or any atom) is. The **buried volume**
            (%V_bur) is the percentage of a sphere centred on the metal that is filled by the atoms of the
            ligands; split into quadrants and octants it shows *where* the bulk is, and the **steric map**
            (a contour plot of the ligand surface) shows the shape of the pocket. Use it to compare ligands
            (NHCs, phosphines, ...), to rationalise selectivity, or to compare conformers of a catalyst.
            The numbers follow the SambVca model (Cavallo and co-workers), so they can be compared with
            literature values computed the same way.
            """,
        needs="""
            * A geometry of the complex: `.xyz`, ORCA or Gaussian output (last geometry). For a multi-structure
              XYZ **only the first structure** is analysed — split ensembles first with `frog-split`.
            * The **0-based** index of the centre (`--center`, usually the metal).
            * The atoms that define the orientation: `-z` (their centroid gives the +Z axis, e.g. the donor
              atom of the ligand you describe) and optionally `-x` (fixes the rotation around Z, needed for
              reproducible quadrants).
            * To describe **one ligand**, the list of all other atoms (other ligands, counter-ions, substrate)
              to leave out with `-E`; `frog-fragments` helps to find the ranges.
            * For maps (`-m`): the Python packages `stericrender` and `xyzrender`; for `--png` also the cairo
              library (macOS: `brew install cairo`).
            """,
        how=[
            "Look up the 0-based indices of the metal, of the donor atom(s) of the ligand you want to describe, "
            "and of all atoms that do not belong to that ligand.",
            "Compute %V_bur: `frog-steric complex.xyz --center 0 -z 1 -x 5 -E 12 13 14 15`. Indices are "
            "listed one by one (ranges like `12-49` are not accepted); in a shell `-E $(seq 12 49)` writes "
            "12 to 49 for you.",
            "Keep the SambVca defaults (sphere radius `-r 3.5` Å, grid `-s 0.1` Å) unless you need to reproduce "
            "a paper that used other values; use the same settings for every structure you compare.",
            "Add `-m` to write `<stem>_steric_map.svg` (and `--png` for a PNG). The map is a view down the Z "
            "axis from +Z; to look at the ligand from the metal side, as steric maps are usually shown, the "
            "ligand must lie on −Z: give its donor atom with `-z` and add `--flip-z`.",
            "If the quadrants come out mirrored or rotated compared with a reference picture, use `--flip-x` or "
            "choose other `-x` atoms; the total %V_bur does not depend on the orientation.",
        ],
        expect="""
            Two tables, one row per file. *Steric Descriptors*: centre, sphere radius (Å), total %V_bur and the
            four quadrants NE, NW, SW, SE (%). *Octant Buried Volume*: the same split into the front (+Z) and
            back (−Z) half. Example (Ru complex `opt.out`, `--center 36 -z 37 -x 3`, nothing excluded):
            %V_bur 96.03 %, quadrants 92.05–98.17 %.

            * With **all** ligands included the sphere is almost full (90–97 % in the examples: Ru complex
              96.0 %, Ag complex `ensemble.xyz` 93.7 %, 85.7 % after excluding the separate 38-atom molecule
              with `-E`). Values for **one ligand**, with all other atoms excluded, are much smaller —
              typically in the 25–50 % range — and are the ones usually reported.
            * Hydrogens are included; radii are Bondi radii × 1.17, and the centre atom itself is never counted.
              Atoms listed with `-E` are excluded from the volume and the map but are still drawn on the map
              (`--no-excluded-on-top` hides them).
            * Quadrants differing by more than ~10–20 % show an asymmetric (e.g. C2- or C1-symmetric) pocket;
              four similar values mean a symmetric environment.
            * With `-m`, if `stericrender`/`xyzrender` are missing, the file is reported as failed and **no
              %V_bur is printed for it**: run again without `-m` to get the numbers. A missing cairo library
              only skips the PNG (warning), the SVG is still written.
            """,
    )
    examples = [
        ('frog-steric complex.xyz --center 0 -z 1 2', '%V_bur around atom 0 with +Z towards atoms 1 and 2'),
        ('frog-steric cat_*.out --center 0 -z 5 -x 7 -E 5 6 -m --png',
         'Steric maps (SVG and PNG) excluding atoms 5 and 6 from the volume'),
    ]

    def setup_args(self, parser):
        parser.add_argument('files', nargs='+', help="Input geometry file(s) (XYZ, ORCA, Gaussian)")
        parser.add_argument('--center', type=int, required=True, metavar='IDX', help="Index of the metal/center atom, 0-based")
        parser.add_argument('-z', '--z-axis', type=int, nargs='+', metavar='IDX',
                            help="Atom indices (0-based) whose centroid defines the +Z axis")
        parser.add_argument('--flip-z', action='store_true', help="Point the Z axis the opposite way (-Z)")
        parser.add_argument('-x', '--x-axis', type=int, nargs='+', metavar='IDX',
                            help="Atom indices (0-based) whose centroid defines the X axis (fixes the roll)")
        parser.add_argument('--flip-x', action='store_true', help="Point the X axis the opposite way (-X)")
        parser.add_argument('-E', '--exclude', type=int, nargs='+', default=[], metavar='IDX',
                            help="Atom indices (0-based) excluded from the volume and the map heights; "
                                 "they are still drawn on the map")
        parser.add_argument('-r', '--radius', type=float, default=3.5, metavar='Å', help="Sphere radius in Å")
        parser.add_argument('-s', '--spacing', type=float, default=0.1, metavar='Å', help="Grid spacing in Å")
        parser.add_argument('-m', '--map', action='store_true', help="Write an SVG steric map for each input")
        parser.add_argument('--no-excluded-on-top', dest='excluded_on_top', action='store_false',
                            help="Do not draw the excluded (-E) atoms on top of the steric map")
        parser.add_argument('--png', action='store_true',
                            help="Also convert the SVG steric map to PNG (requires the cairo library)")
        self.add_out_dir_arg(parser)

    def run(self, args):
        paths = self.require_files(args.files)

        data, octant_rows, failed = [], [], []
        for p in paths:
            print_info(f"Processing {p.name}...")
            try:
                calc = load_calculation(str(p))
            except Exception as e:
                print_error(f"Failed to load {p.name}: {e}")
                failed.append(p.name)
                continue

            atoms = getattr(calc, 'atoms', None)
            coords = getattr(calc, 'geom', None)
            if atoms is None or coords is None:
                print_error(f"No valid geometry found in {p.name}")
                failed.append(p.name)
                continue

            if not 0 <= args.center < len(atoms):
                print_error(f"Center index {args.center} out of range for {p.name} ({len(atoms)} atoms)")
                failed.append(p.name)
                continue

            try:
                oriented_coords = orient_positions(
                    positions=coords,
                    center_idx=args.center,
                    z_axis_idx=args.z_axis,
                    x_axis_idx=args.x_axis,
                    flip_x=args.flip_x,
                    flip_z=args.flip_z,
                )

                # Center and -E atoms are excluded from the volume and the map heights
                # (-E atoms are only drawn on top of the map)
                excluded_set = {args.center, *args.exclude}
                selected = [i for i in range(len(atoms)) if i not in excluded_set]
                selected_positions = oriented_coords[selected]
                radii = get_sambvca_radii([atoms[i] for i in selected])

                vol_res = compute_buried_volume(positions=selected_positions, radii=radii,
                                                radius=args.radius, spacing=args.spacing)
                q, o = vol_res["quadrants"], vol_res["octants"]

                if args.map:
                    self._write_map(p, args, atoms, oriented_coords, selected_positions, radii, vol_res)

            except Exception as e:
                logger.debug("Steric analysis failed", exc_info=True)
                print_error(f"Failed to compute %V_bur for {p.name}: {e}")
                failed.append(p.name)
                continue

            data.append([p.name, f"{atoms[args.center]}{args.center}", float(args.radius),
                         float(vol_res["total"])] + [float(q[k]) for k in QUADRANTS])
            octant_rows.append([p.name] + [float(o[k]) for k in OCTANTS])

        self.output(data, ["File", "Center", "Radius (Å)", "%V_bur"] + [f"{k} (%)" for k in QUADRANTS],
                    title="Steric Descriptors (SambVca Model)", floatfmt='.2f')
        self.output(octant_rows, ["File"] + [f"{k} (%)" for k in OCTANTS],
                    title="Octant Buried Volume", floatfmt='.1f')

        if failed:
            raise InputError(f"{len(failed)} of {len(paths)} file(s) could not be analyzed: {', '.join(failed)}")

    def _write_map(self, p, args, atoms, oriented_coords, selected_positions, radii, vol_res):
        out_svg = self.derived_path(p, 'steric_map', '.svg')
        x_grid, y_grid, z_grid = compute_steric_map(positions=selected_positions, radii=radii,
                                                    radius=args.radius, spacing=args.spacing)
        generate_svg_overlay(
            out_svg=str(out_svg),
            atoms=atoms,
            oriented_coords=oriented_coords,
            radius=args.radius,
            vol_res=vol_res,
            map_x=x_grid,
            map_y=y_grid,
            map_z=z_grid,
            excluded_indices=args.exclude,
            excluded_on_top=args.excluded_on_top,
        )
        self.saved(out_svg, 'steric map')

        if args.png:
            out_png = self.derived_path(p, 'steric_map', '.png')
            try:
                import cairosvg
                cairosvg.svg2png(url=str(out_svg), write_to=str(out_png), scale=2.0)
                self.saved(out_png, 'steric map')
            except Exception as e:
                logging.getLogger("cairocffi").setLevel(logging.CRITICAL)
                logger.debug("SVG to PNG conversion failed: %s", e)
                print_warning("Could not convert the SVG to PNG: the 'cairo' library is missing "
                              "(on macOS: brew install cairo)")


def parse_args():
    return CalcSteric().build_parser().parse_args()

def main():
    CalcSteric().main()

if __name__ == '__main__':
    main()

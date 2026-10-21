"""Compute solvent accessible (SAS) and solvent excluded (SES) surfaces as cube files
from ORCA/Gaussian output or XYZ files.
"""

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.filetypes import load_calculation
from frogchem.printer import print_info
from frogchem.solvent_surface import build_sas_cube, build_ses_cube


def _load_molecule(filepath):
    calc = load_calculation(filepath)
    return calc.atoms, calc.geom


class SolventSurface(CLICommand, name='frog-solvent'):
    """Compute solvent accessible (SAS) and solvent excluded (SES) surfaces as cube files.

    The cubes hold a signed distance field (Å, negative inside the surface): plot the 0.0 isosurface.
    Atomic radii are Bondi vdW radii plus --vdw-offset (default 0).
    """

    category = 'spectroscopy'
    outputs = [
        Output('table', 'Solvent Surface Summary', 'One row per input file: SAS and SES cube paths'),
        Output('file', '<stem>_sas.cube', 'Solvent accessible surface', when='unless --ses-only'),
        Output('file', '<stem>_ses.cube', 'Solvent excluded surface', when='unless --sas-only'),
    ]
    guide = Guide(
        why="""
            Two standard molecular surfaces describe what a solvent "sees". The **solvent accessible surface
            (SAS)** is traced by the centre of a solvent sphere (the *probe*, 1.4 Å ≈ water) rolling on the
            atoms; the **solvent excluded surface (SES**, also called Connolly or molecular surface) is the
            inner surface of the probe, i.e. the boundary of the space the solvent cannot enter. Use them for
            figures (how exposed is a reactive site, a metal centre, a cavity?), to compare the size and shape of
            conformers or catalysts, or to check where a larger solvent molecule fits. This command writes them
            as cube files to be drawn in any viewer.
            """,
        needs="""
            Only a **geometry**: an `.xyz` file or an ORCA/Gaussian output (the last geometry is used); no
            wavefunction is needed. Atomic radii are the Bondi van der Waals radii (H 1.20, C 1.70, N 1.55,
            O 1.52 Å ...), optionally enlarged with `--vdw-offset`; elements missing from the table (e.g. many
            transition metals) get 1.5 × their covalent radius. Choose the probe radius for your solvent with
            `--probe-radius` (default 1.4 Å, water). A molecular viewer that reads cube files (VMD, ChimeraX,
            Avogadro, GaussView) is needed to look at the result.
            """,
        how=[
            "Run `frog-solvent mol.xyz`: writes `mol_sas.cube` and `mol_ses.cube` next to the input "
            "(`--out-dir surfaces` to collect them elsewhere; `--sas-only` or `--ses-only` for one of the two).",
            "Adjust the probe (`--probe-radius 1.9` for a bulkier solvent) and the grid: `--spacing` (Å, "
            "default 0.1) or `--voxels N` per axis, `--padding` (Å, default 3.0).",
            "In the viewer, load the cube together with the molecule and draw the isosurface at value **0.0**.",
        ],
        expect="""
            A *Solvent Surface Summary* table with the paths of the written cubes; with the progress messages
            (hidden by `-q`) each surface also reports its grid and the **enclosed volume** in Å³.

            * The cube values are a **signed distance in Å** from the surface: negative inside, positive outside,
              zero on the surface; so the isovalue to draw is always 0.0 (not a density isovalue).
            * The SAS is much larger than the SES: it lies one probe radius further out. SAS volume ≫ SES volume.
            * Cubes are big at the default 0.1 Å spacing: about 28 MB per cube even for water; for large
              molecules use `--spacing 0.2` or `--voxels 100` for a preview. A coarse grid makes the SES slightly
              bumpy and its volume less accurate.

            Example (`water.xyz`, defaults): 135×126×120 grid; SAS volume 124.50 Å³, SES volume 19.85 Å³,
            two cubes of 28.6 MB each.
            """,
    )
    examples = [
        ('frog-solvent opt.out', 'SAS and SES cubes next to opt.out'),
        ('frog-solvent mol.xyz --sas-only --probe-radius 1.9 --voxels 100',
         'SAS for a larger probe on a 100x100x100 grid'),
        ('frog-solvent conf_*.xyz --out-dir surfaces', 'Surfaces of several conformers in surfaces/'),
    ]

    def setup_args(self, parser):
        parser.add_argument('input', nargs='+', help='ORCA/Gaussian output or XYZ files')
        which = parser.add_mutually_exclusive_group()
        which.add_argument('--sas-only', action='store_true', help='Generate only the SAS cube')
        which.add_argument('--ses-only', action='store_true', help='Generate only the SES cube')
        self.add_grid_args(parser, spacing=0.1, padding=3.0)
        parser.add_argument('--probe-radius', type=float, default=1.4, help='Solvent probe radius in Å')
        parser.add_argument('--vdw-offset', type=float, default=0.0,
                            help='Offset added to the Bondi vdW radius, in Å')
        self.add_out_dir_arg(parser)

    def run(self, args):
        self.require_files(args.input)
        summary_data = []
        for filepath in args.input:
            atoms, coords = _load_molecule(filepath)

            print_info(f'Input: {filepath}')
            if args.voxels:
                print_info(f'Voxels: {args.voxels} per axis, padding: {args.padding} Å')
            else:
                print_info(f'Grid spacing: {args.spacing} Å, padding: {args.padding} Å')
            print_info(f'Probe radius: {args.probe_radius} Å, vdW offset: {args.vdw_offset} Å')

            kwargs = dict(spacing=args.spacing, padding=args.padding, n_voxels=args.voxels,
                          probe_radius=args.probe_radius, vdw_offset=args.vdw_offset)

            sas_out = ses_out = None
            if not args.ses_only:
                sas_out = self.derived_path(filepath, 'sas', '.cube')
                print_info('Computing SAS')
                build_sas_cube(atoms, coords, output=str(sas_out), **kwargs)
                self.saved(sas_out, 'cube')
            if not args.sas_only:
                ses_out = self.derived_path(filepath, 'ses', '.cube')
                print_info('Computing SES')
                build_ses_cube(atoms, coords, output=str(ses_out), **kwargs)
                self.saved(ses_out, 'cube')

            summary_data.append([filepath, str(sas_out) if sas_out else None, str(ses_out) if ses_out else None])

        self.output(summary_data, ['Input File', 'SAS Cube', 'SES Cube'], title='Solvent Surface Summary')


def parse_args():
    return SolventSurface().build_parser().parse_args()


def main():
    SolventSurface().main()


if __name__ == '__main__':
    main()

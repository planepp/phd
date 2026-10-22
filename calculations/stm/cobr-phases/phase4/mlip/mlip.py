#!/usr/bin/env python
"""
Relax one or many structures with a MACE-MP-0 foundation potential.

The model is loaded once, then every folder given is relaxed in turn
(positions only, cell fixed; atoms fixed by Selective dynamics in the
POSCAR stay fixed). In each folder it writes:

  CONTCAR      relaxed structure (VASP format, keeps the constraints)
  mace.traj    optimization trajectory (open with: ase gui mace.traj)
  mace.log     optimizer log
  result.json  energies, steps, convergence, timing

Usage:
  python mlip.py                        # the current folder, as before
  python mlip.py candidates/c0*         # many folders, model loaded once
  python mlip.py candidates/c0* --skip-done
  python mlip.py candidates/c00* --calc emt --fmax 0.05   # quick pipeline test
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS, FIRE


def get_calculator(args):
    if args.calc == 'emt':
        from ase.calculators.emt import EMT
        return EMT()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    from mace.calculators import mace_mp
    return mace_mp(model=args.model, device=args.device, default_dtype=args.dtype)


def relax(folder, calc, args):
    folder = Path(folder)
    atoms = read(folder / args.input, format='vasp')
    atoms.calc = calc
    t0 = time.time()
    e0 = float(atoms.get_potential_energy())
    Opt = {'BFGS': BFGS, 'FIRE': FIRE}[args.optimizer]
    opt = Opt(atoms, trajectory=str(folder / 'mace.traj'), logfile=str(folder / 'mace.log'))
    converged = bool(opt.run(fmax=args.fmax, steps=args.steps))
    e1 = float(atoms.get_potential_energy())
    f = atoms.get_forces(apply_constraint=True)
    write(folder / 'CONTCAR', atoms, format='vasp', direct=True)
    res = dict(E_initial=e0, E_final=e1, converged=converged, steps=int(opt.nsteps),
               fmax_final=float(np.linalg.norm(f, axis=1).max()), time_s=round(time.time() - t0, 1),
               calc=args.calc, model=args.model if args.calc == 'mace' else None,
               dtype=args.dtype if args.calc == 'mace' else None, fmax_target=args.fmax,
               n_atoms=len(atoms))
    (folder / 'result.json').write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='*', default=['.'], help='Folders containing a POSCAR.')
    ap.add_argument('--input', default='POSCAR', help='Structure file name in each folder.')
    ap.add_argument('--calc', choices=['mace', 'emt'], default='mace',
                    help='emt = fast built-in toy potential, only for testing the workflow.')
    ap.add_argument('--model', default='medium', help='MACE-MP model: small / medium / large.')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dtype', default='float64', choices=['float32', 'float64'],
                    help='float64 (default) converges forces reliably; float32 is faster.')
    ap.add_argument('--threads', type=int, default=None, help='Torch threads per process.')
    ap.add_argument('--optimizer', default='BFGS', choices=['BFGS', 'FIRE'])
    ap.add_argument('--fmax', type=float, default=0.01, help='Force criterion (eV/A).')
    ap.add_argument('--steps', type=int, default=500, help='Maximum optimizer steps.')
    ap.add_argument('--skip-done', action='store_true',
                    help='Skip folders that already have a result.json without an error.')
    args = ap.parse_args()

    todo = []
    for f in args.folders:
        f = Path(f)
        if not (f / args.input).exists():
            continue
        rj = f / 'result.json'
        if args.skip_done and rj.exists() and 'error' not in json.loads(rj.read_text()):
            continue
        todo.append(f)
    if not todo:
        print('Nothing to do.'); return

    calc = get_calculator(args)
    print(f'Relaxing {len(todo)} structure(s) with {args.calc}'
          f'{" " + args.model + " " + args.dtype if args.calc == "mace" else ""}')
    n_fail = 0
    for k, f in enumerate(todo, 1):
        try:
            r = relax(f, calc, args)
            flag = '' if r['converged'] else '  NOT CONVERGED'
            print(f'[{k}/{len(todo)}] {f}: E = {r["E_final"]:.4f} eV, {r["steps"]} steps, '
                  f'{r["time_s"]} s{flag}', flush=True)
        except Exception as e:
            n_fail += 1
            (f / 'result.json').write_text(json.dumps({'error': repr(e)}, indent=1))
            print(f'[{k}/{len(todo)}] {f}: FAILED: {e!r}', flush=True)
            traceback.print_exc()
    if n_fail:
        print(f'{n_fail} failed; rerun with --skip-done to retry only those.')
        sys.exit(1)


if __name__ == '__main__':
    main()

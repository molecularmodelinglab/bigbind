from glob import glob
import os
import sys
import openff
from openff.units.openmm import to_openmm
import openmm as mm
from openmm import app
from openmm import unit
from openmmforcefields.generators import EspalomaTemplateGenerator
from openff.toolkit.topology import Molecule
import pandas as pd
from tqdm import tqdm, trange

from baselines.vina_gnina import get_all_bayesbind_struct_splits_and_pockets
from utils.cfg_utils import get_baseline_struct_dir, get_bayesbind_struct_dir, get_config

def minimize_protein(prot, lig, out_file, freeze_alpha=True, anneal=False, include_lig=False, force=False):

    if os.path.exists(out_file) and not force:
        return

    modeller = app.Modeller(app.Topology(), [])
    prot_pdb = app.PDBFile(prot)
    modeller.add(prot_pdb.topology, prot_pdb.positions)

    lig_sdf = Molecule.from_file(lig)
    if isinstance(lig_sdf, list):
        lig_sdf = lig_sdf[0]

    modeller.add(lig_sdf.to_topology().to_openmm(), to_openmm(lig_sdf.conformers[0]))

    topology = modeller.topology
    positions = modeller.positions

    mols = [ lig_sdf ]
    # system_generator = SystemGenerator(forcefields=ffs, small_molecule_forcefield='gaff-2.11', molecules=mols, forcefield_kwargs=forcefield_kwargs, cache='db.json')
    # system = system_generator.create_system(topology)

    espaloma = EspalomaTemplateGenerator(mols, cache="espaloma.json", forcefield='espaloma-0.3.2')
    forcefield = app.ForceField('amber/ff14SB.xml', 'implicit/gbn2.xml')
    forcefield.registerTemplateGenerator(espaloma.generator)
    system = forcefield.createSystem(topology)

    residues = list(topology.residues())
    if freeze_alpha:
        for r in residues:
            for atom in r.atoms():
                if atom.name == "CA":
                    system.setParticleMass(atom.index, 0*unit.amu)

    platform = mm.Platform.getPlatformByName('CUDA')
    # integrator = mm.VerletIntegrator(0.001*unit.picoseconds)
    integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 0.001*unit.picoseconds)
    simulation = app.Simulation(topology, system, integrator, platform)
    context = simulation.context
    context.setPositions(positions)

    # print(f"Initial energy: {context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)} kcal/mol")
    simulation.minimizeEnergy()

    state = context.getState(getPositions=True, getEnergy=True)
    U = state.getPotentialEnergy()
    # print(f"Minimized energy: {U.value_in_unit(unit.kilocalorie_per_mole)} kcal/mol")
    positions = state.getPositions(asNumpy=True)

    if anneal:
        Tmax = 1000*unit.kelvin
        n_iter = 1000
        T = Tmax
        t = trange(n_iter)
        for i in t:
            T = (n_iter-i)*Tmax/n_iter
            integrator.setTemperature(T)
            simulation.step(1000)
            state = simulation.context.getState(getEnergy=True)
            U = state.getPotentialEnergy()
            t.set_description(f"T: {T.value_in_unit(unit.kelvin):0.2f}K, U: {U.value_in_unit(unit.kilocalorie_per_mole):0.2f} kcal/mol")


    if include_lig:
        # save to PDB file (with ligand)
        app.PDBFile.writeFile(topology, positions, open(out_file, 'w'))
    else:
        # save to PDB file (only the protein)
        app.PDBFile.writeFile(prot_pdb.topology, positions[:len(prot_pdb.positions)], open(out_file, 'w'))

def minimize_glide_crossdocked(cfg):

    for split, pocket in (get_all_bayesbind_struct_splits_and_pockets(cfg)):
        if split == "test":
            continue
        rec_file = get_bayesbind_struct_dir(cfg) + f"/{split}/{pocket}/rec_hs.pdb"
        cur_folder = get_baseline_struct_dir(cfg, "glide", split, pocket)
        df = pd.read_csv(get_bayesbind_struct_dir(cfg) + f"/{split}/{pocket}/actives.csv")
        for pdb in tqdm(df.pdb):
            dock_folder = f"{cur_folder}/{pdb}_crossdock"
            # sort so the first one is protein (*-1.pdb)
            docked_pdbs = [ pdb for pdb in sorted(glob(dock_folder + "/*-*.pdb")) if "fixed" not in pdb ]
            if len(docked_pdbs) == 0:
                continue
            assert len(docked_pdbs) > 1
            
            docked_df = pd.read_csv(dock_folder + f"/dock_{pdb}_crossdock.csv")

            _, *ligs = docked_pdbs
            # print(_, ligs)
            for i, (lig, smiles) in enumerate(zip(ligs, docked_df.SMILES)):

                out_file = dock_folder + f"/rec_{i}_full_min.pdb"
                
                # print(rec_file)
                # print(lig)
                try:
                    minimize_protein(rec_file, lig, smiles, out_file)
                except openff.toolkit.utils.exceptions.UndefinedStereochemistryError:
                    # raise
                    print(f"Failed to minimize {out_file}")
                    continue
                # print(f"Saved to {out_file}")

if __name__ == "__main__":
    cfg = get_config(sys.argv[1])
    minimize_glide_crossdocked(cfg)
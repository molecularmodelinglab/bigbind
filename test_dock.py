from rdkit import Chem
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from vina import Vina
v = Vina(sf_name='vina')
 
v.set_receptor('test_rec.pdbqt')
v.set_ligand_from_file('test_lig.pdbqt')
 
v.compute_vina_maps(center=[-9, 3, 12], box_size=[10, 10, 10])
 
energy_minimized = v.optimize()
print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
v.write_pose('test_minimized.pdbqt', overwrite=True)

n_poses = 100
# Dock the ligand
v.dock(exhaustiveness=32, n_poses=n_poses, min_rmsd=1)
v.write_poses('test_out.pdbqt', n_poses=n_poses, overwrite=True)

# The BigBind Dataset and BayesBind Benchmark

Here lies the code for generating the [BigBind Dataset](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.3c01211) and [BayesBind virtual screening benchmark](https://arxiv.org/abs/2403.10478v1). Download the latest BigBind Dataset (V1.5) [here](https://storage.googleapis.com/bigbind_data/BigBindV1.5.tar.gz) and the latest BayesBind benchmark (V1.5, full set) [here](https://storage.googleapis.com/bigbind_data/BayesBindV1.5.tar.gz). The BayesBind ML subset is [here](https://storage.googleapis.com/bigbind_data/BayesBindMLV1.5.tar.gz).

## BigBind contents

The BigBind dataset has the following directory structure:

```
BidBindV1.5
├── {pocket_name}/
│   ├── {rec_pdb_id}_rec.pdb
│   ├── {rec_pdb_id}_rec_pocket.pdb
│   └── {lig_pdb_id}_lig.sdf
├── ... (many pockets)
├── chembl_structures/
│   └── mol_${mol_number}.sdf
├── structures_{all|train|val|test}.csv
├── activities_{all|train|val|test}.csv
├── activities_sna_1_{train|val|test}.csv
└── {val|test}_screens/
    └── {pocket_name}.csv
```

There's a bunch of pocket folders with fancy names like `FABF_BACSU_1_413_0`. Within each folder, there's a bunch of aligned pdb files. The `*_rec.pdb` files are the full receptor, and the `*_rec_pocket.pdb` files are just the pocket residues (residues within 5 Å of any crystallized ligand). Each crystallized ligand also has its own sdf file. In the paper, the whole receptor files and ligand files are never used, but they are nonetheless present for posterity.

The main meat of the dataset is in the `activities_*.csv` files. The non-SNA activities csvs contain all the (filtered) ChEMBL activities, and have the following columns:

| Column           | Description                                                    |
| :---             |    :----                                                       |
| lig_smiles       | SMILES string for the ligand                                   |
| lig_file         | filename of UFF-optimized ligand sdf (in `chembl_structures/`) |
| standard_type    | type of the activity from ChEMBL. E.g. IC50 or Ki/Kd           |
| standard_relation | relation from ChEMBL (=, >, <). Always = in BigBind            |
| standard_value   | raw value of the activity                                      |
| standard_units   | units of the raw activity (nM, μM, etc). Will usually be nM but plz don't rely on that |
| pchembl_value    | ChEMBL-provided log-normalized activity value. Defined at -log_10(activity in nM) |
| active           | whether of not the pchembl value is greater than our activity cutoff (10 μM) |
| uniprot          | Uniprot accession of the receptor |
| pocket           | Receptor pocket folder. We assume the ligand can bind to any of the aligned receptors in the folder |
| ex_rec_file      | a randomly selected full receptor from the pocket folder |
| ex_rec_pdb       | the pdb id of that randomly selected receptor |
| ex_rec_pocket_file | the pdb file with just the pocket residues for the selected receptor |
| num_pocket_residues | number of residues in the example pocket file |
| pocket_center_{x,y,z} | center of the example pocket bounding box |
| pocket_size_{x,y,z}   | sizes of the example pocket bounding box |
| lig_cluster           | Ligand Tanimoto cluster ID (0.4 cutoff) for a particular pocket. Cluster IDs are not globally unique; they are only unique within a pocket  |
| rec_cluster           | Receptor pocket cluster (based on pocket-TM-score) | 

The SNA activities csvs (`activities_sna_1_*.csv`) have the same structure, but are augmented with putative inactive compounds with no labelled activity.

There are also `structures_*.csv` files describing the 3d crystal structures of ligands if you so desire. This data isn't used in the paper, but you might find it useful. It's really just a way to access the CrossDocked data if you don't care about the docked poses. The structure of the structures file is as follows:

| Column           | Description                                                    |
| :---             |    :----                                                       |
| pdb              | pdb id of the crystal structure the ligand came from |
| pocket           | same as in activities                                   |
| lig_smiles       | same as in activities                                   |
| lig_crystal_file | filename of the ligand sdf in its crystal 3D pose |
| lig_uff_file     | filename of UFF-optimized ligand. Useful for e.g. docking |
| redock_rec_*         | Receptor data where rec pdb == lig pdb |
| crossdocked_rec_*    | Receptor data selected from pocket where rec pdb != lig pdb |
| num_pocket_residues | same as in activities                                   |
| pocket_center_{x,y,z} | same as in activities                                   |
| pocket_size_{x,y,z}   | same as in activities                                   |

## All dataset versions

| Version | Notes | ChEMBL version |
| :---    | :---  | :---           |
| [1.5](https://storage.googleapis.com/plantain_data/BigBindV1.5.tar.gz)     | More rigorous data splits via pocket-TM-score. Also removed noisy HTS data | 33 |
| [1.0](https://storage.googleapis.com/bigbind/BigBindV1.tar.bz2) | Initial version of the dataset. Used ProBis splits | 30 |

# The BayesBind benchmark

The benchmark directory structure is as follows:
```
BayesBindV1
└── "val" | "test"
    └── {pocket_name}/
        ├── rec.pdb
        ├── pocket.pdb
        ├── actives.smi
        ├── actives.csv
        ├── random.smi
        └── random.csv
```

For each pocket in the benchmark, `rec.pdb` and `pocket.pdb` are the structures of the full receptor and just the pocket of the receptor, respectively. We have separate csv files for the random and active set (each csv file follows the same format as the BigBind dataset above; since each file is pocket-specific, a lot of the columns are the same value). For convenience, there are also smi files for both sets containing just the SMILES of each compound.

**Warining!** The name "actives" can be misleading -- the csv and smi files contain molecules with *measured activities*, but these activities can be below our cutoff. In the BayesBind paper, we use a `pchembl` cutoff of 5 (though others can be chosen). This means all compounds with activities below the cutoff are discarded when computing EF<sup>B</sup>s. I'm aware this is pretty confusing nomenclature -- please let me know if you have better ideas.

A reference implementation of the EF<sup>B</sup> metric is in `baselines/efb.py`.

If you are the first to achieve a median EF<sup>B</sup><sub>max</sub> of > 50 on targets in the BayesBind test set (either ML or full), please [let me know](mailto:mixarcid@unc.edu)! I will personally buy drinks for every member of your group.

## Creating the dataset and benchmark

Right now the code for creating the dataset is relatively brittle and specific to the machines we use. We are working on making this more usable. In the meantime, please reach out to [mixarcid@unc.edu](mailto:mixarcid@unc.edu) if you want to get this running yourself. I am happy to help!

# The BigBind Dataset

Here lies the code for generating the BigBind Dataset. I've also documented the dataset contents here.

## Dataset contents

If you download the dataset [here](https://drive.google.com/file/d/15D6kQZM0FQ2pgpMGJK-5P9T12ZRjBjXS/view?usp=sharing) and extract it, you'll find the following directory structure:

```
BidBindV1
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

There's a bunch of pocket folders with fancy names like `FABF_BACSU_1_413_0`. Within each folder there's a bunch of aligned pdb files. The `*_rec.pdb` files are the full receptor, and the `*_rec_pocket.pdb` files are just the pocket residues (residues within 5 Å of any crystallized ligand). Each crystallized ligand also has its own sdf file. In the paper, the whole receptor files and ligand files are never used, but they are nonetheless present for posterity.

The main meat of the dataset is in the `activities_*.csv` files. The non-SNA activities csvs contain all the (filtered) ChEMBL activities, and have the following columns:

| Column           | Description                                                    |
| :---             |    :----                                                       |
| lig_smiles       | SMILES string for the ligand                                   |
| lig_file         | filename of UFF-optimized ligand sdf (in `chembl_structures/`) |
| standard_type    | type of the activity from ChEMBL. E.g. IC50 of Ki/Kd           |
| standard_relation | relation from ChEMBL (=, >, <). Always = in BigBind            |
| standard_value   | raw value of the activity                                      |
| standard_units   | units of the raw activity (nM, μM, etc). Will usually be nM but plz don't rely on that |
| pchembl_value    | ChEMBL-provided log-normalized activity value. Defined at -log_10(activity in nM) |
| active           | whether of not the pchembl value is greater than our activity cutoff (10 μM) |
| uniprot          | Uniprot accession of the receptor |
| pocket           | Receptor pocket folder. We assume the ligand can bind to any of the aligned recptors in the folder |
| ex_rec_file      | a randomly selected full receptor from the pocket folder |
| ex_rec_pdb       | the pdb id of that randomly selected receptor |
| ex_rec_pocket_file | the pdb file with just the pocket residues for the selected receptor |
| num_pocket_residues | number of residues in the example pocket file |
| pocket_center_{x,y,z} | center of the example pocket bounding box |
| pocket_size_{x,y,z}   | sizes of the example pocket bounding box |

The SNA activities csvs (`activities_sna_1_*.csv`) have the same structure, but don't have any of the specific activity-related values (e.g. `pchembl_value`). Instead all they have is the boolean `active` column.

The `*_screens` folders contain the virtual screening benchmarks described in the paper. There is a seperate csv for each pocket, structured exactly like the SNA csv files.

There are also `structures_*.csv` files describing the 3d crystal structures of ligands if you so desire. This data isn't used in the paper, but you might find it useful. It's really just a way to access the CrossDocked data if you don't care about the docked poses. The structure of the structues file is as follows:

| Column           | Description                                                    |
| :---             |    :----                                                       |
| lig_smiles       | same as in activities                                   |
| lig_file         | filename of the ligand sdf in its crystal 3D pose |
| lig_pdb          | pdb id of the crystal structure the ligand came from |
| pocket           | same as in activities                                   |
| ex_rec_*         | same as in activities. To support cross-docking structure prediction, we assert ex_rec_pdb != lig_pdb |
| num_pocket_residues | same as in activities                                   |
| pocket_center_{x,y,z} | same as in activities                                   |
| pocket_size_{x,y,z}   | same as in activities                                   |

## Creating the dataset

If you desire to create the dataset yourself, first you'll need to get the input data. Download and extract the [ChEMBL sqlite database](https://chembl.gitbook.io/chembl-interface-documentation/downloads) (version 30 in the paper), the [CrossDocked dataset](http://bits.csb.pitt.edu/files/crossdock2020/), [LIT-PCBA](https://drugdesign.unistra.fr/LIT-PCBA/), and the [SIFTS database](https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html) (you want the `pdb_chain_uniprot.csv` file).

Now you create your config file. Put the following into `cfg.yaml`:

```yaml

bigbind_folder: "/path/to/desired/output/folder"
crossdocked_folder: "/path/to/CrossDocked/dataset"
lit_pcba_folder: "/path/to/LIT-PCBA/dataset"
chembl_file: "/path/to/chembl/sqlite/dataset"
sifts_file: "/path/to/SIFTS/pdb_chain_uniprot/csv/file"

# we cache all intermediate outputs. This is where we should save the cached files
cache_folder: "/path/to/cache/folder"
cache: true
# if something went wrong and you need to recalculate a cached function,
# put the function name in the recalc list
recalc:
  - null

```

Now pip install the requirements, and you're ready to run. Simply run `python run.py` and it will produce the dataset. Since we cache intermediate outputs, if something goes wrong you can re-run `run.py` and it will start where it left off. Just make sure you modify offending functions and add them to the `recalc` list before re-running.

Enjoy! If you have any questions, email me at [mixarcid@unc.edu](mailto:mixarcid@unc.edu)
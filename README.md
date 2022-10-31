# The BigBind Dataset

Here lies the code for generating the BigBind Dataset. I've also documented the dataset contents here.

## Dataset contents

If you download the dataset [here](https://drive.google.com/file/d/15D6kQZM0FQ2pgpMGJK-5P9T12ZRjBjXS/view?usp=sharing) and extract it, you'll find the following directory structure:

```
BidBindV1
├── ${pocket_name}/
│   ├── ${rec_pdb_id}_rec.pdb
│   ├── ${rec_pdb_id}_rec_pocket.pdb
│   └── ${lig_pdb_id}_lig.sdf
├── ... (many pockets)
├── chembl_structures/
│   └── mol_${mol_number}.sdf
├── structures_${all|train|val|test}.csv
├── activities_${all|train|val|test}.csv
├── activities_sna_1_${train|val|test}.csv
└── {val|test}_screens/
    └── ${pocket_name}.csv
```

There's a bunch of pocket folders with fancy names like `FABF_BACSU_1_413_0`. Within each folder there's a bunch of aligned pdb files. The `*_rec.pdb` files are the full receptor, and the `*_rec_pocket.pdb` files are just the pocket residues (residues within 5 Å of any crystallized ligand). Each crystallized ligand also has its own sdf file. In the paper, the whole receptor files and ligand files are never used, but they are nonetheless present for posterity.

The main meat of the dataset is in the `activities_*.csv` files. The non-SNA activities csvs contain all the (filtered) ChEMBL activities, and have the following columns:

| Column        | Description                                                    |
| :---          |    :----:                                                      |
| lig_smiles    | SMILES string for the ligand                                   |
| lig_file      | filename of UFF-optimized ligand sdf (in `chembl_structures/`) |

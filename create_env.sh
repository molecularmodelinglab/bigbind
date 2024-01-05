mamba create -n bigbind -y &&
conda activate bigbind &&
mamba install -c conda-forge -c omnia openmm openff-toolkit openff-forcefields openmmforcefields espaloma=0.3.2 -y &&
pip install -r requirements.txt
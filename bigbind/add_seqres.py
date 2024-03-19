import requests
import io

def add_seq_to_pdb(in_file, pdb_id, out_file):
    """ Adds SEQRES to a PDB based on the PDB ID """

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    r = requests.get(url)
    og_lines = io.StringIO(r.content.decode("utf-8")).readlines()

    seqres_lines = [ line for line in og_lines if line.startswith("SEQRES") ]
    
    pdb_lines = seqres_lines + open(in_file).readlines()
    with open(out_file, "w") as f:
        f.write("".join(pdb_lines))

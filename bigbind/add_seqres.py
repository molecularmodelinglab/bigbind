import re
from Bio import SeqIO
from Bio.SeqUtils import seq3
from Bio.PDB import PDBParser
import requests
import io

# def download_sequence(pdb_id):
#     # Download sequence in FASTA format from RCSB PDB
#     url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
#     r = requests.get(url)
#     buf = io.StringIO(r.content.decode("utf-8"))
#     print(url)

#     return list(SeqIO.parse(buf, "fasta"))

# def make_seqres_block(chain_id, seq):
#     """" Returns a SEQRES block from the chain id (e.g. A)
#     and the peptide sequence as a list of lines """
#     length = len(seq)
#     lines = []
#     cur_line = None

#     for res in seq:
#         if cur_line is None or len(cur_line) == 70:
#             if cur_line is not None:
#                 # lmao apparently these extra spaces really matter
#                 lines.append(cur_line + "          \n")

#             ser_num = len(lines) + 1
#             cur_line = f"SEQRES  "
#             cur_line += f"{ser_num:>2} {chain_id} {length:>4} "
#         cur_line += " " + seq3(res).upper()

#     lines.append(cur_line + "          \n")
#     return lines


# auth_regex = re.compile("(([^ ]+)\[auth ([^ ])\])")
# def add_seq_to_pdb_old(in_file, pdb_id, out_file):
#     """ Downloads the sequences for the pdb_id and adds them to the pdb file,
#     saving as out_file """

#     seqs = download_sequence(pdb_id.upper())

#     chain2seq = {}
#     for seq in seqs:
#         chain_str = seq.description.split("|")[1]

#         # CrossDocked uses author chain names. We use
#         # this crused regex solution to convert them
#         for match, _, auth_name in auth_regex.findall(chain_str):
#             chain_str = chain_str.replace(match, auth_name)

#         _, *chains = chain_str.split()
#         for chain in chains:
#             chain = chain.replace(",", "")
#             chain2seq[chain] = seq

#     parser = PDBParser(QUIET=True)
#     structure = parser.get_structure("protein", in_file)


#     pdb_lines = []
#     for i, chain in enumerate(structure[0]):
#         seq = chain2seq[chain.id]
#         seqres_block = make_seqres_block(chain.id, seq)
#         pdb_lines += seqres_block

#     pdb_lines += open(in_file).readlines()
#     with open(out_file, "w") as f:
#         f.write("".join(pdb_lines))

def add_seq_to_pdb(in_file, pdb_id, out_file):
    """ Much simpler and more effect way to add seqres blocks """

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    r = requests.get(url)
    og_lines = io.StringIO(r.content.decode("utf-8")).readlines()

    seqres_lines = [ line for line in og_lines if line.startswith("SEQRES") ]
    
    pdb_lines = seqres_lines + open(in_file).readlines()
    with open(out_file, "w") as f:
        f.write("".join(pdb_lines))

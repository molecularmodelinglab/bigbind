import ssl
import urllib.request
import os
from task import Task

# todo: disable. Something is up with chembl certs and this is the only fix rn
ssl._create_default_https_context = ssl._create_unverified_context

class StaticDownloadTask(Task):
    """ Assumes file to download doesn't change and thus only needs to
    be downloaded onces. File is only downloaded locally """

    def __init__(self, name, url):
        rel_out_fname = os.path.basename(url)
        super().__init__(name, rel_out_fname, False)
        self.url = url

    def run(self, cfg):
        # https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
        filename = self.get_out_filename(cfg)
        print(f"Writing to {filename}")
        urllib.request.urlretrieve(self.url, filename)

download_chembl = StaticDownloadTask("download_chembl", "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_33_mysql.tar.gz")
download_sifts = StaticDownloadTask("download_sifts", "ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz")
download_crossdocked = StaticDownloadTask("download_crossdocked", "https://storage.googleapis.com/plantain_data/CrossDocked2022.tar.gz")
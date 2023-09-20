import ssl
import shutil
import urllib.request
import os
from utils.task import Task

# todo: disable. Something is up with chembl certs and this is the only fix rn
ssl._create_default_https_context = ssl._create_unverified_context

class StaticDownloadTask(Task):
    """ Assumes file to download doesn't change and thus only needs to
    be downloaded onces. File is only downloaded locally """

    def __init__(self, name, url):
        rel_out_fname = os.path.basename(url)
        super().__init__(name, rel_out_fname, local=True)
        self.url = url

    def run(self, cfg, prev_output=None):
        filename = self.get_out_filename(cfg)
        # if prev_output is not None:
        #     print(f"Using previous data from {self.name}")
        #     shutil.copyfile(prev_output, filename)
        print(f"Writing to {filename}")
        urllib.request.urlretrieve(self.url, filename)
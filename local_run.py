import sys
from omegaconf import OmegaConf

def test_f():
    import time
    time.sleep(60)
    with open("test.txt", "w") as f:
        f.write("WORKING\n")

NAME_TO_FUNC = {
    "test_f": test_f
}

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/hosts.yaml")
    host_name = sys.argv[1]
    func_name = sys.argv[2]

    host = cfg[host_name]
    func = NAME_TO_FUNC[func_name]
    func()
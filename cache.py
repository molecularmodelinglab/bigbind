import pickle
import os
import yaml

def cache(f):
    def cached_f(cfg, *args, **kwargs):
        if cfg["cache"] and cfg["recalc"] != f.__name__:
            cache_file = f"{cfg['cache_folder']}/{f.__name__}.pkl"
            try:
                with open(cache_file, "rb") as fh:
                    ret = pickle.load(fh)
                    print(f"Using cached data from {f.__name__}")
                    return ret
            except:
                print(f"Running {f.name}")
                ret = f(cfg, *args, **kwargs)
                with open(cache_file, "wb") as fh:
                    pickle.dump(ret, fh)
                return ret
    return cached_f
        
num_times_called = 0
@cache
def test_cache(cfg):
    global num_times_called
    num_times_called += 1
    return "working"
    
if __name__ == "__main__":
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cache_file = f"{cfg['cache_folder']}/test_cache.pkl"
    try:
        os.remove(cache_file)
    except FileNotFoundError:
        pass
    
    t1 = test_cache(cfg)
    t2 = test_cache(cfg)
    assert os.path.exists(cache_file)
    assert num_times_called == 1
    

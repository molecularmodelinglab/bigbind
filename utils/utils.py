from typing import Dict, Any, Callable

def recursive_map(func: Callable, item: Any):
    """ recurisvely applies func to all non-container
    items in the container """
    
    if isinstance(item, list) or isinstance(item, tuple):
        return type(item)([ recursive_map(func, i2) for i2 in item ])
    elif isinstance(item, dict):
        return { key: recursive_map(func, val) for key, val in item.items() }
    else:
        return func(item)
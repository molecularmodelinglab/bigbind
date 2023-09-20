from typing import Dict, Any, Callable
import inspect

def recursive_map(func: Callable, item: Any):
    """ recurisvely applies func to all non-container
    items in the container """
    
    if isinstance(item, list) or isinstance(item, tuple):
        return type(item)([ recursive_map(func, i2) for i2 in item ])
    elif isinstance(item, dict):
        return { key: recursive_map(func, val) for key, val in item.items() }
    else:
        return func(item)

def can_take_kwarg(fn, argname):
    """ Returns true if fn can take a kwarg with name argname """
    spec = inspect.getfullargspec(fn)
    if argname in spec.args:
        return True
    elif spec.varkw is not None:
        return True
    return False

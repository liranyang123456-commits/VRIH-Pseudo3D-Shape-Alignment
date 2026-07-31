import sys

import h5py


with h5py.File(sys.argv[1], "r") as handle:
    names = list(handle.keys())
    print(names[:10])
    if names:
        node = handle[names[0]]
        print(type(node), getattr(node, "shape", None))
        if isinstance(node, h5py.Group):
            print(list(node.keys()))

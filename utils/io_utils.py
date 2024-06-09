import pickle
import numpy as np
from plyfile import PlyData, PlyElement

def read_pickle(fname):
    try:
        with open(fname, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"[Error] {e}")

        import pickle5
        with open(fname, 'rb') as f:
            data_dict = pickle5.load(f)

    return data_dict

def write_pickle(fname, data):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
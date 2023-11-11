import numpy as np
import argparse

def load_tet(file:str):
    """Load a .tet file into a numpy array.
    N vertices
    M cells
    x y z # vertices
    4 i j k l # indices of vertices
    Args:
        file (str): Path to the .tet file.

    Returns:
        verts (np.ndarray): A numpy array containing the vertices of the mesh.
        cells (np.ndarray): A numpy array containing the cells of the mesh.
    """
    with open(file, 'r') as f:
        lines = f.readlines()
        num_vertices = int(lines[0].split()[0])
        num_cells = int(lines[1].split()[0])
        verts = np.zeros((num_vertices, 3))
        cells = np.zeros((num_cells, 4), dtype=int)
        for i, line in enumerate(lines[2:]):
            if i < num_vertices:
                verts[i] = np.array([float(x) for x in line.split()])
            else:
                cells[i-num_vertices] = np.array([int(x) for x in line.split()])[1:5]
    return verts, cells

def write_tet(file:str, verts:np.ndarray, cells:np.ndarray):
    """Write a .tet file from a numpy array.
    N vertices
    M cells
    x y z # vertices
    4 i j k l # indices of vertices
    Args:
        file (str): Path to the .tet file.
        verts (np.ndarray): A numpy array containing the vertices of the mesh.
        cells (np.ndarray): A numpy array containing the cells of the mesh.
    """
    with open(file, 'w') as f:
        f.write(f'{verts.shape[0]} vertices\n')
        f.write(f'{cells.shape[0]} cells\n')
        for vert in verts:
            f.write(f'{vert[0]} {vert[1]} {vert[2]}\n')
        for cell in cells:
            f.write(f'4 {cell[0]} {cell[1]} {cell[2]} {cell[3]}\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Normalize a .tet file.')
    parser.add_argument('input_file', type=str, help='Path to the input .tet file.')
    parser.add_argument('output_file', type=str, help='Path to the output .tet file.')
    return parser.parse_args()

def normalize_tet(verts, scale = 1000):
    """Normalize a tet mesh to fit in a unit cube. The origin is placed at left bottom corner.
    Args:
        verts (np.ndarray): A numpy array containing the vertices of the mesh.
        scale (float): The scale to normalize the mesh to.
    Returns:
        verts (np.ndarray): A numpy array containing the vertices of the mesh.
    """
    xmin = np.min(verts[:, 0])
    ymin = np.min(verts[:, 1])
    zmin = np.min(verts[:, 2])
    xmax = np.max(verts[:, 0])
    ymax = np.max(verts[:, 1])
    zmax = np.max(verts[:, 2])
    maxside = max(max(xmax - xmin, ymax - ymin), zmax - zmin)
    verts[:, 0] = (verts[:, 0] - xmin) / maxside * scale
    verts[:, 1] = (verts[:, 1] - ymin) / maxside * scale
    verts[:, 2] = (verts[:, 2] - zmin) / maxside * scale

if __name__ == '__main__':
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file
    print("input file: ", input_file)
    print("output file: ", output_file)

    verts, cells = load_tet(input_file)
    normalize_tet(verts)
    write_tet(output_file, verts, cells)


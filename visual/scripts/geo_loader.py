import numpy as np

def load_xyz(file:str):
    """Load a .xyz file into a numpy array.

    Args:
        file (str): Path to the .xyz file.

    Returns:
        np.ndarray: A numpy array containing the vertices of the mesh.
    """
    with open(file, 'r') as f:
        lines = f.readlines()
        num_vertices = int(lines[0])
        vertices = np.zeros((num_vertices, 3))
        for i, line in enumerate(lines[1:]):
            vertices[i] = np.array([float(x) for x in line.split()])
    return vertices

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

class GeoLoader:
    def __init__(self, geo):
        self.geo = geo
    
    def load_xyz(self, file:str):
        """Load a .xyz file into a houdini geo.

        Args:
            file (str): Path to the .xyz file.

        Returns:
            hou.Geometry: A houdini geo containing the vertices of the mesh.
        """
        vertices = load_xyz(file)
        for v in vertices:
            self.geo.createPoint().setPosition(v)
        return self.geo
    
    def load_tet(self, file:str):
        verts, cells = load_tet(file)
        verts, cells = load_tet(file)
        points = self.geo.createPoints(verts)
        for c in cells:
            triangles = np.array([[c[0], c[1], c[2]], [c[0], c[2], c[3]], [c[0], c[3], c[1]], [c[1], c[3], c[2]]])
            for t in triangles:
                poly = self.geo.createPolygon()
                for i in t:
                    poly.addVertex(points[i])
        return self.geo

if __name__ == "__main__":
    verts, indices = load_tet('../../CMakeBuild/data/joint.tet')
    print(verts.shape)
    print(indices.shape)



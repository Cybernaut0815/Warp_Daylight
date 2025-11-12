import trimesh
import numpy as np
from typing import Tuple


def load_obj(file_path: str) -> trimesh.Trimesh:
    """Loads a mesh from a file and returns a trimesh object.
    
    Args:
        file_path: The path to the OBJ file.
        
    Returns:
        A trimesh object.
    """
    mesh = trimesh.load(file_path)
    return mesh


def save_obj(mesh: trimesh.Trimesh, file_path: str) -> None:
    """Saves a mesh to a file.
    
    Args:
        mesh: The mesh to save.
        file_path: The path to the OBJ file.
    """
    mesh.export(file_path)


def load_vertices_faces_indices(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads a mesh from a file and returns the vertices, faces, and indices.
    
    Args:
        file_path: The path to the OBJ file.
        
    Returns:
        A tuple containing the vertices, faces, and indices.
    """
    mesh = load_obj(file_path)
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Face counts: number of vertices per face (numpy array)
    face_counts = np.array([len(face) for face in faces], dtype=np.int32)
    
    # Face indices: flattened array of vertex indices
    face_indices = faces.flatten().astype(np.int32)
    
    return vertices, face_counts, face_indices
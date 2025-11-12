from pxr import Vt, Sdf, Usd, UsdGeom
from typing import Tuple, List, Optional
import numpy as np


def load_usd(file_path: str, prim_path: Optional[str] = None) -> Usd.Stage:
    """Loads a USD file and returns the stage.
    
    Args:
        file_path: The path to the USD file.
        prim_path: Optional specific prim path. If None, returns the stage.
        
    Returns:
        The USD stage.
    """
    stage = Usd.Stage.Open(file_path)
    return stage


def get_first_mesh(stage: Usd.Stage) -> Optional[UsdGeom.Mesh]:
    """Returns the first mesh found in the USD stage.
    
    Args:
        stage: The USD stage.
        
    Returns:
        The first UsdGeom.Mesh found, or None if no mesh exists.
    """
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            return UsdGeom.Mesh(prim)
    return None


def load_vertices_faces_indices(file_path: str, prim_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads a USD file and returns the vertices, face counts, and face indices.
    
    Args:
        file_path: The path to the USD file.
        prim_path: Optional specific prim path to a mesh. If None, uses the first mesh found.
        
    Returns:
        A tuple containing the vertices, face counts, and face indices.
    """
    stage = load_usd(file_path)
    
    if prim_path:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
    else:
        mesh = get_first_mesh(stage)
        if mesh is None:
            raise ValueError(f"No mesh found in USD file: {file_path}")
    
    # Get vertices
    points_attr = mesh.GetPointsAttr()
    vertices = np.array(points_attr.Get())
    
    # Get face vertex counts
    face_counts_attr = mesh.GetFaceVertexCountsAttr()
    face_counts = np.array(face_counts_attr.Get())
    
    # Get face vertex indices
    face_indices_attr = mesh.GetFaceVertexIndicesAttr()
    face_indices = np.array(face_indices_attr.Get())
    
    return vertices, face_counts, face_indices
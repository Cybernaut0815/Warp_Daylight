import trimesh
import numpy as np

def visualize_mesh(vertices: np.ndarray, 
                    face_counts: np.ndarray, 
                    faces_indices: np.ndarray, 
                    vertex_colors: np.ndarray = None, 
                    vertex_normals: np.ndarray = None, 
                    face_normals: np.ndarray = None,
                    wireframe: bool = True,
                    line_width: float = 1.0) -> None:
    """
    Visualize a mesh with optional vertex colors and normals.
    
    Args:
        vertices: Nx3 array of vertex positions
        face_counts: Array of vertex counts per face
        faces_indices: Flattened array of face vertex indices
        vertex_colors: Optional Nx3 or Nx4 array of vertex colors (RGB or RGBA)
        vertex_normals: Optional Nx3 array of vertex normals
        face_normals: Optional Mx3 array of face normals (M = number of faces)
        wireframe: Whether to show wireframe edges (default: True)
        line_width: Width of lines for wireframe and normals (default: 1.0)
    """
    # Convert face_counts and faces_indices to trimesh faces format
    faces = []
    offset = 0
    for count in face_counts:
        face = faces_indices[offset:offset + count]
        faces.append(face)
        offset += count
    
    faces = np.array(faces)
    
    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # Apply vertex colors if provided
    if vertex_colors is not None:
        mesh.visual.vertex_colors = vertex_colors
    
    # Create scene with the mesh
    scene = trimesh.Scene(mesh)
    scene.set_camera(angles=(0, 0, 0))
    
    # Add wireframe using edges if enabled
    if wireframe:
        edges = mesh.edges_unique
        edge_vertices = vertices[edges]
        wireframe_path = trimesh.load_path(edge_vertices)
        # Set line width via entity metadata
        for entity in wireframe_path.entities:
            entity.color = [50, 50, 50, 255]  # Dark gray
        wireframe_path.metadata['line_width'] = line_width
        scene.add_geometry(wireframe_path)
    
    # Add vertex normals as arrows if provided
    if vertex_normals is not None:
        normal_length = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) * 0.05
        segments = np.hstack([
            vertices,
            vertices + vertex_normals * normal_length
        ]).reshape(-1, 2, 3)
        vertex_normal_vectors = trimesh.load_path(segments)
        vertex_normal_vectors.colors = np.array([[0, 255, 0, 255]] * len(vertices))  # Green for vertex normals
        vertex_normal_vectors.metadata['line_width'] = line_width
        scene.add_geometry(vertex_normal_vectors)
    
    # Add face normals as arrows if provided
    if face_normals is not None:
        face_centers = np.array([vertices[face].mean(axis=0) for face in faces])
        normal_length = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) * 0.05
        segments = np.hstack([
            face_centers,
            face_centers + face_normals * normal_length
        ]).reshape(-1, 2, 3)
        face_normal_vectors = trimesh.load_path(segments)
        face_normal_vectors.colors = np.array([[255, 0, 0, 255]] * len(faces))  # Red for face normals
        face_normal_vectors.metadata['line_width'] = line_width
        scene.add_geometry(face_normal_vectors)
    
    # Show the scene with line width configuration
    scene.show(line_settings={'point_size': line_width * 3, 'line_width': line_width})
"""
Utility functions for 3D visualization
"""
import numpy as np
import trimesh
from typing import Tuple


def create_scene_with_mesh_and_wireframe(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray = None,
    face_colors: np.ndarray = None,
    wireframe_color: Tuple[int, int, int, int] = (80, 80, 80, 255),
    line_width: float = 1.0
) -> trimesh.Scene:
    """
    Create a trimesh Scene with a mesh and optional wireframe.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        vertex_colors: Optional vertex colors (N, 3) or (N, 4)
        face_colors: Optional face colors (M, 3) or (M, 4) - takes precedence over vertex_colors
        wireframe_color: RGBA color for wireframe edges
        line_width: Width of wireframe lines
    
    Returns:
        scene: trimesh Scene object
    """
    # Create mesh
    mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # Set face colors if provided (takes precedence)
    if face_colors is not None:
        mesh_obj.visual.face_colors = face_colors
    # Otherwise set vertex colors if provided
    elif vertex_colors is not None:
        mesh_obj.visual.vertex_colors = vertex_colors
    
    # Create scene
    scene = trimesh.Scene(mesh_obj)
    
    # Add wireframe
    edges = mesh_obj.edges_unique
    edge_vertices = vertices[edges]
    wireframe_path = trimesh.load_path(edge_vertices)
    
    # Set wireframe color on entities
    for entity in wireframe_path.entities:
        entity.color = list(wireframe_color)
    
    scene.add_geometry(wireframe_path)
    
    return scene


def add_sun_vectors_to_scene(
    scene: trimesh.Scene,
    light_directions: np.ndarray,
    distance: float = 0.5,
    length: float = 0.1,
    color: Tuple[int, int, int, int] = (255, 200, 0, 255)
) -> None:
    """
    Add sun direction vectors to a trimesh Scene as yellow/orange arrows.
    
    Args:
        scene: trimesh Scene to add vectors to
        light_directions: Light direction vectors FROM sun (N, 3)
        distance: Distance from origin to place sun positions
        length: Length of the arrow vectors
        color: RGBA color for the vectors
    """
    # Sun positions are opposite of light directions (light comes FROM sun)
    sun_positions = -light_directions
    
    for sun_pos in sun_positions:
        # Position at distance from origin (where the sun is)
        start_pos = sun_pos * distance
        
        # Arrow points toward the scene to show light direction
        end_pos = start_pos - sun_pos * length
        
        # Create individual path for this vector
        segment = np.array([[start_pos, end_pos]])
        sun_vector_path = trimesh.load_path(segment)
        
        # Set color
        for entity in sun_vector_path.entities:
            entity.color = list(color)
        
        scene.add_geometry(sun_vector_path)


def create_manual_arc_sun_vectors(
    num_samples: int = 50,
    arc_angle_range: Tuple[float, float] = (-60, 60),
    arc_rotation_deg: float = 45
) -> np.ndarray:
    """
    Create sun direction vectors in a manually defined arc above the model.
    This is useful as a fallback when Ladybug is not available.
    
    Args:
        num_samples: Number of sun position samples
        arc_angle_range: Tuple of (min_angle, max_angle) in degrees for the arc
        arc_rotation_deg: Rotation angle around X-axis in degrees
    
    Returns:
        light_directions: Light direction vectors FROM sun (N, 3)
    """
    sun_angles = np.linspace(arc_angle_range[0], arc_angle_range[1], num_samples)
    arc_rotation_rad = np.radians(arc_rotation_deg)
    
    light_directions = []
    
    for angle_deg in sun_angles:
        angle_rad = np.radians(angle_deg)
        
        # Sun position: x-component varies with angle, y is up, z is forward
        sun_pos = np.array([
            np.sin(angle_rad),  # x: varies from -1 to 1
            np.cos(angle_rad),  # y: always positive (above)
            0.0                  # z: no forward/backward component
        ])
        
        # Rotate around X-axis to tilt the arc along its diameter
        cos_rot = np.cos(arc_rotation_rad)
        sin_rot = np.sin(arc_rotation_rad)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_rot, -sin_rot],
            [0, sin_rot, cos_rot]
        ])
        
        sun_pos = rotation_matrix @ sun_pos
        sun_pos = sun_pos / np.linalg.norm(sun_pos)  # Normalize
        
        # Light comes FROM sun (negative of position)
        light_directions.append(-sun_pos)
    
    return np.array(light_directions, dtype=np.float32)


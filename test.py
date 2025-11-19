from re import L
import warp as wp
import numpy as np
import requests
import os
import trimesh

from src.IO import load_obj
from src.viz import mesh
from src.raycast import direct_sunlight

if __name__ == "__main__":
    print("CUDA Device Count:")
    print("--------------------------------")
    print(wp.get_cuda_device_count())

    if not os.path.exists("data/meshes/bunny.obj"):
        print("Stanford Bunny not found, downloading...")
        response = requests.get("https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj")
        with open("data/meshes/bunny.obj", "wb") as f:
            f.write(response.content)
        print("\nStanford Bunny downloaded successfully.\n")
    else:
        print("\nStanford Bunny already downloaded. Skipping download.\n")
        
    vertices, face_counts, face_indices = load_obj.load_vertices_faces_indices("data/meshes/bunny.obj")
    
    print("Vertices:")
    print("--------------------------------")
    print(vertices.shape)
    print("Face Counts:")
    print("--------------------------------")
    print(face_counts.shape)
    print("Face Indices:")
    print("--------------------------------")
    print(face_indices.shape)
    
    # Reshape face_indices into faces array (n, 3) for triangular meshes
    faces = face_indices.reshape(-1, 3)
    
    # Compute face normals using cross product
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Face normals via cross product
    face_normals = np.cross(edge1, edge2)
    # Normalize
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10)
    
    # Compute vertex normals using trimesh.geometry.mean_vertex_normals
    vertex_normals = trimesh.geometry.mean_vertex_normals(
        vertex_count=len(vertices),
        faces=faces,
        face_normals=face_normals
    )
    
    print("\nVertex Normals:")
    print("--------------------------------")
    print(vertex_normals.shape)
    print(vertex_normals[:10])
    
    
    ###### Raycasting ######
    ###### NEEDS TESTING AND OPTIMIZATION ######
    
    start_positions = vertices.copy()
    start_normals = vertex_normals.copy()
    
    ### Raycasting ###
    hitcounts = direct_sunlight.raycast_directional(vertices, face_indices, start_positions, start_normals, ray_directions)
    print("\nHitcounts:")
    print("--------------------------------")
    print(hitcounts.shape)
    print(hitcounts[:10])
    
    ###### End of Raycasting ######
    
    
    ###### Visualizing ######
    ###### For showing the raycasting results ######
    
    mesh.visualize_mesh(vertices, 
                        face_counts, 
                        face_indices, 
                        vertex_colors=None, 
                        vertex_normals=vertex_normals, 
                        face_normals=face_normals, 
                        wireframe=True,
                        line_width=1.0)
    
    # mesh.visualize_mesh(vertices, 
    #                 face_counts, 
    #                 face_indices, 
    #                 vertex_colors=None, 
    #                 vertex_normals=vertex_normals, 
    #                 face_normals=face_normals, 
    #                 wireframe=True,
    #                 line_width=1.0)
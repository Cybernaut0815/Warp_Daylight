"""
Shared sample payloads for API tests.

The mesh is a simple unit-square plane made of two triangles, facing +Y.
"""

SIMPLE_MESH = {
    "points": [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ],
    "face_vertex_counts": [3, 3],
    "face_vertex_indices": [0, 1, 2, 0, 2, 3],
    "normals": [
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
}

SIMPLE_SUN_CONFIG = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "timezone": "America/New_York",
    "coordinate_system": "y_up",
    "start_datetime": "2024-06-21T10:00:00",
    "end_datetime": "2024-06-21T14:00:00",
    "time_step_minutes": 60,
}

DEFAULT_OPTIONS = {
    "offset_distance": 0.001,
    "chunk_size": 0,
    "use_backface_culling": True,
    "return_colors": True,
}

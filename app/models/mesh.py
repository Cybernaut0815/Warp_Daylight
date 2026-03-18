"""
Pydantic models for USD-like mesh data.

These models mirror the UsdGeom.Mesh schema so DCC tools (Maya, Blender,
Omniverse, Rhino) can send mesh data using familiar attribute names.
"""
from pydantic import BaseModel, Field, model_validator
from typing import Optional


class MeshData(BaseModel):
    """
    USD-like mesh representation.

    Follows the UsdGeom.Mesh convention:
      - points:              [[x,y,z], ...] vertex positions
      - face_vertex_counts:  [3, 3, 4, ...] number of vertices per face
      - face_vertex_indices:  flat list of vertex indices consumed per face
      - normals:             optional per-vertex normals [[nx,ny,nz], ...]

    Example (two triangles sharing an edge):
        {
            "points": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
            "face_vertex_counts": [3, 3],
            "face_vertex_indices": [0,1,2, 0,2,3],
            "normals": null
        }
    """
    points: list[list[float]] = Field(
        ...,
        description="Vertex positions as [[x,y,z], ...]. "
                    "Each inner list must have exactly 3 elements.",
        min_length=3,
    )
    face_vertex_counts: list[int] = Field(
        ...,
        description="Number of vertices per face, e.g. [3,3,4]. "
                    "Supported values: 3 (triangle) and 4 (quad).",
        min_length=1,
    )
    face_vertex_indices: list[int] = Field(
        ...,
        description="Flat list of vertex indices consumed by each face. "
                    "Length must equal sum(face_vertex_counts).",
        min_length=3,
    )
    normals: Optional[list[list[float]]] = Field(
        default=None,
        description="Optional per-vertex normals [[nx,ny,nz], ...]. "
                    "If omitted, normals are computed automatically.",
    )

    @model_validator(mode="after")
    def validate_mesh_consistency(self) -> "MeshData":
        for i, pt in enumerate(self.points):
            if len(pt) != 3:
                raise ValueError(
                    f"points[{i}] has {len(pt)} components, expected 3"
                )

        expected_index_count = sum(self.face_vertex_counts)
        if len(self.face_vertex_indices) != expected_index_count:
            raise ValueError(
                f"face_vertex_indices length ({len(self.face_vertex_indices)}) "
                f"!= sum(face_vertex_counts) ({expected_index_count})"
            )

        num_verts = len(self.points)
        for i, idx in enumerate(self.face_vertex_indices):
            if idx < 0 or idx >= num_verts:
                raise ValueError(
                    f"face_vertex_indices[{i}] = {idx} is out of range "
                    f"[0, {num_verts})"
                )

        for i, count in enumerate(self.face_vertex_counts):
            if count < 3 or count > 4:
                raise ValueError(
                    f"face_vertex_counts[{i}] = {count}; "
                    "only triangles (3) and quads (4) are supported"
                )

        if self.normals is not None:
            if len(self.normals) != num_verts:
                raise ValueError(
                    f"normals length ({len(self.normals)}) != "
                    f"points length ({num_verts})"
                )
            for i, n in enumerate(self.normals):
                if len(n) != 3:
                    raise ValueError(
                        f"normals[{i}] has {len(n)} components, expected 3"
                    )

        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "points": [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    "face_vertex_counts": [3, 3],
                    "face_vertex_indices": [0, 1, 2, 0, 2, 3],
                    "normals": None,
                }
            ]
        }
    }

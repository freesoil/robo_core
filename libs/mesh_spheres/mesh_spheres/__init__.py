"""
Mesh to Spheres Conversion Library

Converts STL meshes to collections of spheres using octree-based adaptive subdivision.
Each cubic cell is represented as a circumsphere (diagonal radius) for complete coverage.
"""

from .converter import MeshToSpheresConverter, SphereCollection, Sphere

__all__ = [
    'MeshToSpheresConverter',
    'SphereCollection',
    'Sphere',
]

__version__ = '2.0.0'

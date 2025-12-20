"""
Visualization utilities for mesh and sphere collections.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from typing import Optional
from ..converter import SphereCollection


def visualize_spheres(collection: SphereCollection, 
                     mesh: Optional[trimesh.Trimesh] = None,
                     show_mesh: bool = True,
                     alpha: float = 0.3,
                     figsize: tuple = (12, 8)):
    """
    Visualize sphere collection with optional mesh overlay.
    
    Args:
        collection: SphereCollection to visualize
        mesh: Optional mesh to overlay
        show_mesh: Whether to show mesh wireframe
        alpha: Transparency of spheres
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot spheres
    colors = plt.cm.viridis(np.linspace(0, 1, len(collection)))
    
    for sphere, color in zip(collection.spheres, colors):
        # Create sphere mesh
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = sphere.center[0] + sphere.radius * np.outer(np.cos(u), np.sin(v))
        y = sphere.center[1] + sphere.radius * np.outer(np.sin(u), np.sin(v))
        z = sphere.center[2] + sphere.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')
    
    # Plot mesh wireframe if provided
    if mesh is not None and show_mesh:
        # Sample edges for wireframe
        edges = mesh.edges_unique
        edge_points = mesh.vertices[edges]
        
        for edge in edge_points:
            ax.plot3D(*edge.T, 'k-', linewidth=0.5, alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Sphere Approximation ({len(collection)} spheres)')
    
    # Equal aspect ratio
    min_bounds, max_bounds = collection.bounds()
    max_range = (max_bounds - min_bounds).max()
    center = (min_bounds + max_bounds) / 2
    
    ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
    ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
    ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
    
    plt.tight_layout()
    return fig, ax


def visualize_2d_projection(collection: SphereCollection,
                           camera_matrix: np.ndarray,
                           image_size: tuple,
                           background: Optional[np.ndarray] = None,
                           figsize: tuple = (10, 8)):
    """
    Visualize 2D projection of spheres onto image plane.
    
    Args:
        collection: SphereCollection to project
        camera_matrix: 3x4 camera projection matrix
        image_size: (width, height)
        background: Optional background image
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Show background if provided
    if background is not None:
        ax.imshow(background, extent=[0, image_size[0], image_size[1], 0])
    else:
        ax.set_xlim(0, image_size[0])
        ax.set_ylim(image_size[1], 0)
    
    # Project spheres
    projections = collection.project_to_2d(camera_matrix, image_size)
    
    # Draw circles
    colors = plt.cm.viridis(np.linspace(0, 1, len(projections)))
    
    for (center, radius), color in zip(projections, colors):
        circle = plt.Circle(center, radius, color=color, fill=True, 
                          alpha=0.5, edgecolor='white', linewidth=1)
        ax.add_patch(circle)
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'2D Projection ({len(projections)} visible spheres)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig, ax


def compare_methods(mesh_path: str, figsize: tuple = (18, 6)):
    """
    Compare different conversion methods side-by-side.
    
    Args:
        mesh_path: Path to STL file
        figsize: Figure size
    """
    from ..converter import MeshToSpheresConverter
    
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Test different methods
    methods = ['adaptive', 'hierarchical', 'medial_axis']
    configs = [
        {'method': method, 'target_spheres': 50}
        for method in methods
    ]
    
    fig = plt.figure(figsize=figsize)
    
    for idx, (method, config) in enumerate(zip(methods, configs)):
        converter = MeshToSpheresConverter(config)
        collection = converter.convert(mesh_path)
        
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # Plot spheres
        colors = plt.cm.viridis(np.linspace(0, 1, len(collection)))
        
        for sphere, color in zip(collection.spheres, colors):
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = sphere.center[0] + sphere.radius * np.outer(np.cos(u), np.sin(v))
            y = sphere.center[1] + sphere.radius * np.outer(np.sin(u), np.sin(v))
            z = sphere.center[2] + sphere.radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x, y, z, color=color, alpha=0.3, edgecolor='none')
        
        # Plot mesh edges
        edges = mesh.edges_unique
        edge_points = mesh.vertices[edges]
        for edge in edge_points[::5]:  # Subsample for speed
            ax.plot3D(*edge.T, 'k-', linewidth=0.3, alpha=0.2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{method}\n{len(collection)} spheres\n'
                    f'Coverage: {collection.metadata["coverage_ratio"]:.2%}')
        
        # Equal aspect ratio
        min_bounds, max_bounds = collection.bounds()
        max_range = (max_bounds - min_bounds).max()
        center = (min_bounds + max_bounds) / 2
        
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
    
    plt.tight_layout()
    return fig


def plot_coverage_vs_spheres(mesh_path: str, 
                            method: str = 'adaptive',
                            max_spheres: int = 200,
                            step: int = 20):
    """
    Plot coverage ratio vs number of spheres to find optimal configuration.
    """
    from ..converter import MeshToSpheresConverter
    
    sphere_counts = range(10, max_spheres + 1, step)
    coverages = []
    
    for n in sphere_counts:
        config = {'method': method, 'target_spheres': n}
        converter = MeshToSpheresConverter(config)
        collection = converter.convert(mesh_path)
        coverages.append(collection.metadata['coverage_ratio'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(sphere_counts, coverages, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Number of Spheres')
    plt.ylabel('Coverage Ratio')
    plt.title(f'Coverage vs Number of Spheres ({method})')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% coverage')
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()





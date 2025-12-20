"""
Algorithms for mesh-to-spheres conversion.
"""

import numpy as np
from typing import List
import trimesh
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans


def adaptive_sampling(mesh: trimesh.Trimesh, 
                     target_spheres: int,
                     min_radius: float, 
                     max_radius: float,
                     coverage_threshold: float) -> List:
    """
    Adaptive greedy sphere placement algorithm.
    
    Strategy:
    1. Sample dense points on mesh surface
    2. Iteratively place spheres to maximize coverage
    3. Use varying radii based on local geometry
    4. Stop when coverage threshold or target count reached
    
    Best for: General purpose, good balance of accuracy and efficiency
    """
    from .converter import Sphere
    
    # Check for valid mesh
    if mesh.area <= 0 or len(mesh.vertices) == 0:
        # Return single sphere at centroid
        center = mesh.centroid if hasattr(mesh, 'centroid') else np.array([0, 0, 0])
        radius = min_radius
        return [Sphere(center=center, radius=radius)]
    
    # Sample points on surface (dense sampling)
    n_samples = max(5000, int(mesh.area * 1000))
    n_samples = min(n_samples, 50000)  # Cap at 50k for performance
    
    try:
        surface_points, face_indices = trimesh.sample.sample_surface(
            mesh, n_samples
        )
    except Exception as e:
        # Fallback if sampling fails
        print(f"Warning: Surface sampling failed ({e}), using vertices")
        surface_points = mesh.vertices
        if len(surface_points) == 0:
            center = np.array([0, 0, 0])
            return [Sphere(center=center, radius=min_radius)]
    
    # Track uncovered points
    uncovered_mask = np.ones(len(surface_points), dtype=bool)
    spheres = []
    
    # Build KDTree for efficient nearest neighbor queries
    kdtree = KDTree(surface_points)
    
    while len(spheres) < target_spheres:
        uncovered_points = surface_points[uncovered_mask]
        
        if len(uncovered_points) == 0:
            break
        
        # Check coverage
        coverage = 1.0 - (uncovered_mask.sum() / len(surface_points))
        if coverage >= coverage_threshold:
            break
        
        # Find densest uncovered region
        # Use clustering on uncovered points
        if len(uncovered_points) < 10:
            center = uncovered_points.mean(axis=0)
        else:
            # Find point with most neighbors
            n_neighbors = min(50, len(uncovered_points))
            uncovered_kdtree = KDTree(uncovered_points)
            densities = np.array([
                len(uncovered_kdtree.query_ball_point(pt, max_radius))
                for pt in uncovered_points
            ])
            center_idx = densities.argmax()
            center = uncovered_points[center_idx]
        
        # Determine optimal radius for this sphere
        # Start with max radius and shrink if needed
        nearby_indices = kdtree.query_ball_point(center, max_radius)
        nearby_uncovered = surface_points[nearby_indices][uncovered_mask[nearby_indices]]
        
        if len(nearby_uncovered) > 0:
            # Use distance to furthest uncovered point within max_radius
            distances = np.linalg.norm(nearby_uncovered - center, axis=1)
            radius = min(max_radius, np.percentile(distances, 85))
            radius = max(min_radius, radius)
        else:
            radius = min_radius
        
        # Add sphere
        sphere = Sphere(center=center, radius=radius)
        spheres.append(sphere)
        
        # Update coverage mask
        distances = np.linalg.norm(surface_points - center, axis=1)
        newly_covered = distances <= radius
        uncovered_mask &= ~newly_covered
    
    return spheres


def hierarchical_decomposition(mesh: trimesh.Trimesh,
                              target_spheres: int,
                              min_radius: float,
                              max_radius: float) -> List:
    """
    Hierarchical octree-based sphere decomposition.
    
    Strategy:
    1. Voxelize the mesh
    2. Build octree structure
    3. Fit spheres to leaf nodes
    4. Merge small spheres when beneficial
    
    Best for: Simple objects, fast computation, fewer spheres
    """
    from .converter import Sphere
    
    # Check for valid mesh
    if len(mesh.vertices) == 0:
        center = np.array([0, 0, 0])
        return [Sphere(center=center, radius=min_radius)]
    
    # Check for degenerate bounds
    bounds_size = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    if bounds_size == 0 or not np.isfinite(bounds_size):
        center = mesh.centroid if hasattr(mesh, 'centroid') else np.array([0, 0, 0])
        return [Sphere(center=center, radius=min_radius)]
    
    # Voxelize mesh - adaptive resolution based on target spheres
    voxel_size = bounds_size / (target_spheres ** (1/3) * 2)
    voxel_size = max(voxel_size, min_radius)
    
    voxels = mesh.voxelized(pitch=voxel_size)
    voxel_centers = voxels.points
    
    if len(voxel_centers) == 0:
        # Fallback: use mesh center
        center = mesh.centroid
        radius = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) / 4
        return [Sphere(center=center, radius=radius)]
    
    # Cluster voxels
    n_clusters = min(target_spheres, len(voxel_centers))
    
    if n_clusters < len(voxel_centers):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(voxel_centers)
        cluster_centers = kmeans.cluster_centers_
    else:
        labels = np.arange(len(voxel_centers))
        cluster_centers = voxel_centers
    
    # Create spheres for each cluster
    spheres = []
    for i in range(len(cluster_centers)):
        cluster_points = voxel_centers[labels == i]
        
        if len(cluster_points) == 0:
            continue
        
        # Sphere center is cluster center
        center = cluster_centers[i]
        
        # Radius is distance to furthest point + voxel size
        distances = np.linalg.norm(cluster_points - center, axis=1)
        radius = distances.max() + voxel_size / 2
        
        # Clamp radius
        radius = np.clip(radius, min_radius, max_radius)
        
        spheres.append(Sphere(center=center, radius=radius))
    
    return spheres


def medial_axis_approximation(mesh: trimesh.Trimesh,
                             min_radius: float,
                             max_radius: float,
                             sample_density: int) -> List:
    """
    Medial axis transform approximation.
    
    Strategy:
    1. Sample points inside and on surface of mesh
    2. Compute distance to surface for interior points
    3. Extract medial axis (skeleton) points
    4. Place spheres along medial axis
    
    Best for: Collision detection, follows object skeleton
    """
    from .converter import Sphere
    
    # Get mesh bounds
    bounds_min, bounds_max = mesh.bounds
    bounds_size = bounds_max - bounds_min
    
    # Create grid for distance field
    grid_resolution = int(np.cbrt(sample_density))
    grid_resolution = max(20, min(100, grid_resolution))
    
    # Create 3D grid
    x = np.linspace(bounds_min[0], bounds_max[0], grid_resolution)
    y = np.linspace(bounds_min[1], bounds_max[1], grid_resolution)
    z = np.linspace(bounds_min[2], bounds_max[2], grid_resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    
    # Check which points are inside mesh
    inside_mask = mesh.contains(grid_points)
    inside_points = grid_points[inside_mask]
    
    if len(inside_points) == 0:
        # Fallback: use surface sampling
        return _fallback_surface_sampling(mesh, min_radius, max_radius, 30)
    
    # Compute distance to surface for inside points
    # Use proximity query to mesh surface
    closest_points, distances, _ = trimesh.proximity.closest_point(
        mesh, inside_points
    )
    
    # Find local maxima in distance field (medial axis approximation)
    # Reshape distances to grid
    distance_grid = np.zeros(xx.shape)
    inside_indices = np.where(inside_mask)[0]
    
    for idx, dist in zip(inside_indices, distances):
        grid_idx = np.unravel_index(idx, xx.shape)
        distance_grid[grid_idx] = dist
    
    # Find local maxima
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(distance_grid, size=3)
    medial_mask = (distance_grid == local_max) & (distance_grid > 0)
    
    # Extract medial axis points and their distances (radii)
    medial_indices = np.where(medial_mask.ravel())[0]
    medial_points = grid_points[medial_indices]
    medial_radii = distance_grid.ravel()[medial_indices]
    
    # Filter by radius constraints
    valid_mask = (medial_radii >= min_radius) & (medial_radii <= max_radius)
    medial_points = medial_points[valid_mask]
    medial_radii = medial_radii[valid_mask]
    
    if len(medial_points) == 0:
        return _fallback_surface_sampling(mesh, min_radius, max_radius, 30)
    
    # Sort by radius (larger first) for better coverage
    sort_indices = np.argsort(-medial_radii)
    medial_points = medial_points[sort_indices]
    medial_radii = medial_radii[sort_indices]
    
    # Create spheres, removing redundant ones
    spheres = []
    for center, radius in zip(medial_points, medial_radii):
        # Check if this sphere is redundant
        is_redundant = False
        for existing_sphere in spheres:
            # If center is inside existing sphere, skip
            if existing_sphere.contains_point(center):
                is_redundant = True
                break
        
        if not is_redundant:
            spheres.append(Sphere(center=center, radius=float(radius)))
    
    # If we have too few spheres, add surface coverage
    if len(spheres) < 5:
        additional = _fallback_surface_sampling(mesh, min_radius, max_radius, 10)
        spheres.extend(additional)
    
    return spheres


def _fallback_surface_sampling(mesh: trimesh.Trimesh, 
                               min_radius: float,
                               max_radius: float,
                               n_spheres: int) -> List:
    """Fallback method using simple surface sampling."""
    from .converter import Sphere
    
    # Check for valid mesh
    if len(mesh.vertices) == 0:
        return [Sphere(center=np.array([0, 0, 0]), radius=min_radius)]
    
    # Sample points on surface
    try:
        surface_points, _ = trimesh.sample.sample_surface(mesh, n_spheres * 10)
    except:
        # Use vertices if sampling fails
        surface_points = mesh.vertices
        if len(surface_points) == 0:
            return [Sphere(center=np.array([0, 0, 0]), radius=min_radius)]
    
    # Cluster into n_spheres groups
    if len(surface_points) > n_spheres:
        kmeans = KMeans(n_clusters=n_spheres, random_state=42, n_init=10)
        labels = kmeans.fit_predict(surface_points)
        centers = kmeans.cluster_centers_
    else:
        centers = surface_points
    
    # Create spheres
    spheres = []
    for center in centers:
        # Radius based on distance to nearest surface point
        distances = np.linalg.norm(surface_points - center, axis=1)
        radius = np.clip(np.percentile(distances, 50), min_radius, max_radius)
        spheres.append(Sphere(center=center, radius=float(radius)))
    
    return spheres



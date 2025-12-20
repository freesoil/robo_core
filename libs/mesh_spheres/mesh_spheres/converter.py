"""
Main converter class for mesh-to-spheres conversion.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import trimesh
from pathlib import Path


@dataclass
class Sphere:
    """Represents a single sphere in 3D space."""
    center: np.ndarray  # [x, y, z]
    radius: float
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside the sphere."""
        return np.linalg.norm(point - self.center) <= self.radius
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate distance from sphere surface to point."""
        return np.linalg.norm(point - self.center) - self.radius
    
    def intersects_sphere(self, other: 'Sphere') -> bool:
        """Check if this sphere intersects with another."""
        distance = np.linalg.norm(self.center - other.center)
        return distance <= (self.radius + other.radius)
    
    def volume(self) -> float:
        """Calculate sphere volume."""
        return (4/3) * np.pi * (self.radius ** 3)


@dataclass
class SphereCollection:
    """Collection of spheres representing a mesh."""
    spheres: List[Sphere] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.spheres)
    
    def total_volume(self) -> float:
        """Calculate total volume of all spheres."""
        return sum(s.volume() for s in self.spheres)
    
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of all spheres."""
        if not self.spheres:
            return np.zeros(3), np.zeros(3)
        
        centers = np.array([s.center for s in self.spheres])
        radii = np.array([s.radius for s in self.spheres])
        
        min_bounds = (centers - radii[:, None]).min(axis=0)
        max_bounds = (centers + radii[:, None]).max(axis=0)
        
        return min_bounds, max_bounds
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside any sphere."""
        return any(s.contains_point(point) for s in self.spheres)
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """Get minimum distance from point to any sphere surface."""
        if not self.spheres:
            return float('inf')
        return min(s.distance_to_point(point) for s in self.spheres)
    
    def project_to_2d(self, camera_matrix: np.ndarray, 
                      image_size: Tuple[int, int]) -> List[Tuple[np.ndarray, float]]:
        """
        Project spheres to 2D image plane.
        
        Args:
            camera_matrix: 3x4 camera projection matrix
            image_size: (width, height) of output image
            
        Returns:
            List of (center_2d, radius_2d) tuples for each visible sphere
        """
        projections = []
        
        for sphere in self.spheres:
            # Project center
            center_3d_h = np.append(sphere.center, 1)  # Homogeneous coordinates
            center_2d_h = camera_matrix @ center_3d_h
            
            if center_2d_h[2] <= 0:  # Behind camera
                continue
                
            center_2d = center_2d_h[:2] / center_2d_h[2]
            
            # Approximate radius projection
            # Project a point on sphere surface to estimate 2D radius
            offset = np.array([sphere.radius, 0, 0])
            surface_point_h = np.append(sphere.center + offset, 1)
            surface_2d_h = camera_matrix @ surface_point_h
            
            if surface_2d_h[2] > 0:
                surface_2d = surface_2d_h[:2] / surface_2d_h[2]
                radius_2d = np.linalg.norm(surface_2d - center_2d)
                
                # Check if within image bounds
                if (0 <= center_2d[0] < image_size[0] and 
                    0 <= center_2d[1] < image_size[1]):
                    projections.append((center_2d, radius_2d))
        
        return projections
    
    def save(self, filepath: str):
        """Save sphere collection to file."""
        import pickle
        
        # Convert spheres to simple arrays
        centers = np.array([s.center for s in self.spheres])
        radii = np.array([s.radius for s in self.spheres])
        
        # Save with numpy for arrays, pickle for metadata
        np.savez(
            filepath,
            centers=centers,
            radii=radii,
            num_spheres=len(self.spheres)
        )
        
        # Save metadata separately using pickle (handles nested structures)
        metadata_file = str(filepath).replace('.npz', '_metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'SphereCollection':
        """Load sphere collection from file."""
        import pickle
        from pathlib import Path
        
        # Load arrays
        data = np.load(filepath, allow_pickle=True)
        centers = data['centers']
        radii = data['radii']
        
        # Reconstruct spheres
        spheres = [Sphere(center=centers[i], radius=float(radii[i])) 
                   for i in range(len(centers))]
        
        # Load metadata
        # Match the save method's filename construction
        metadata_file = str(filepath).replace('.npz', '_metadata.pkl')
        metadata_file_path = Path(metadata_file)
        if metadata_file_path.exists():
            with open(metadata_file_path, 'rb') as f:
                metadata = pickle.load(f)
        else:
            # Fallback for old format
            metadata = data['metadata'].item() if 'metadata' in data else {}
        
        return cls(spheres=spheres, metadata=metadata)


class MeshToSpheresConverter:
    """
    Converts STL meshes to sphere collections using octree-based adaptive subdivision.
    
    Uses hierarchical octree to subdivide space until coverage and precision criteria are met.
    Each cubic cell is represented as a circumsphere (diagonal radius) ensuring complete coverage.
    """
    
    # Optimized configuration for good mesh approximation
    DEFAULT_CONFIG = {
        'min_radius_ratio': 0.012,  # Min radius as fraction of mesh size (1.2% - fine detail)
        'max_radius_ratio': 0.30,  # Max radius as fraction of mesh size (30% - efficient bulk)
        'coverage_threshold': 0.95,  # 95% surface coverage (high completeness)
        'precision_threshold': 0.70,  # 70% mesh occupancy (good fit, minimizes empty space)
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize converter with configuration.
        
        Args:
            config: Configuration dict. Uses DEFAULT_CONFIG if None.
        """
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)
    
    def convert(self, mesh_path: str) -> SphereCollection:
        """
        Convert STL mesh to sphere collection.
        
        Args:
            mesh_path: Path to STL file
            
        Returns:
            SphereCollection representing the mesh
        """
        # Load mesh
        mesh = trimesh.load(mesh_path)
        
        # Get mesh properties
        mesh_size = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        
        # Convert radius ratios to absolute values
        # But allow override with absolute values if provided
        if 'min_radius_absolute' in self.config:
            min_radius = self.config['min_radius_absolute']
        else:
            min_radius = mesh_size * self.config['min_radius_ratio']
            
        if 'max_radius_absolute' in self.config:
            max_radius = self.config['max_radius_absolute']
        else:
            max_radius = mesh_size * self.config['max_radius_ratio']
        
        # Use octree-based adaptive subdivision (only method)
        spheres = self._octree_adaptive(
            mesh,
            min_radius,
            max_radius,
            self.config['coverage_threshold'],
            self.config['precision_threshold']
        )
        
        # Calculate comprehensive metrics
        metrics = self._calculate_coverage_and_accuracy(mesh, spheres)
        
        # Create collection with metadata
        collection = SphereCollection(
            spheres=spheres,
            metadata={
                'source_file': str(mesh_path),
                'method': 'octree_adaptive',
                'config': self.config,
                'mesh_volume': mesh.volume,
                'mesh_bounds': mesh.bounds.tolist(),
                'num_spheres': len(spheres),
                'coverage_ratio': metrics['coverage_ratio'],
                'precision_ratio': metrics['precision_ratio'],
                'mean_error': metrics['mean_error'],
                'rms_error': metrics['rms_error'],
                'max_error': metrics['max_error'],
                'efficiency': metrics['efficiency']
            }
        )
        
        return collection
    
    def _octree_adaptive(self, mesh: trimesh.Trimesh,
                        min_radius: float,
                        max_radius: float,
                        coverage_threshold: float,
                        precision_threshold: float) -> List[Sphere]:
        """
        Octree-based adaptive sphere generation (criteria-driven).
        
        Subdivides space hierarchically until coverage and precision targets are met.
        Each cubic cell is represented as a circumsphere (diagonal radius).
        """
        from .octree_adaptive import octree_adaptive_spheres
        return octree_adaptive_spheres(
            mesh, min_radius, max_radius, coverage_threshold, precision_threshold
        )
    
    def _calculate_coverage_and_accuracy(self, mesh: trimesh.Trimesh, 
                                        spheres: List[Sphere]) -> Dict[str, float]:
        """
        Calculate both coverage and accuracy metrics for sphere approximation.
        
        Metrics:
        - coverage_ratio: % of mesh surface covered by spheres (recall)
        - precision_ratio: % of sphere volume that contains mesh (minimize empty space)
        - mean_error: average distance from surface to nearest sphere
        - rms_error: root mean square error
        - max_error: worst-case error
        - efficiency: coverage per sphere
        
        Returns:
            Dict with all metrics
        """
        # Check for invalid mesh
        if mesh.area <= 0 or len(mesh.vertices) == 0:
            return {
                'coverage_ratio': 1.0 if len(spheres) == 0 else 0.0,
                'precision_ratio': 0.0,
                'mean_error': 0.0,
                'rms_error': 0.0,
                'max_error': 0.0,
                'efficiency': 0.0
            }
        
        # Sample points on mesh surface
        n_samples = max(100, min(10000, int(mesh.area * 100)))
        
        try:
            points, _ = trimesh.sample.sample_surface(mesh, n_samples)
        except Exception:
            points = mesh.vertices
        
        if len(points) == 0:
            return {
                'coverage_ratio': 1.0 if len(spheres) == 0 else 0.0,
                'precision_ratio': 0.0,
                'mean_error': 0.0,
                'rms_error': 0.0,
                'max_error': 0.0,
                'efficiency': 0.0
            }
        
        # Calculate coverage and errors
        covered = 0
        errors = []
        
        for point in points:
            # Find distance to nearest sphere surface
            min_distance = float('inf')
            point_covered = False
            
            for sphere in spheres:
                dist = sphere.distance_to_point(point)
                abs_dist = abs(dist)
                
                if abs_dist < min_distance:
                    min_distance = abs_dist
                
                if dist <= 0:  # Inside sphere
                    point_covered = True
            
            if point_covered:
                covered += 1
            
            # Track approximation error (distance to nearest sphere surface)
            errors.append(min_distance)
        
        errors_array = np.array(errors)
        coverage_ratio = covered / len(points)
        mean_error = np.mean(errors_array)
        rms_error = np.sqrt(np.mean(errors_array ** 2))
        max_error = np.max(errors_array)
        
        # Calculate precision: mesh occupancy inside spheres
        # Sample points inside sphere volumes and check if they're in the mesh
        precision_ratio = self._calculate_precision(mesh, spheres)
        
        # Efficiency: coverage per sphere (higher is better)
        efficiency = coverage_ratio / len(spheres) if len(spheres) > 0 else 0.0
        
        return {
            'coverage_ratio': float(coverage_ratio),
            'precision_ratio': float(precision_ratio),
            'mean_error': float(mean_error),
            'rms_error': float(rms_error),
            'max_error': float(max_error),
            'efficiency': float(efficiency)
        }
    
    def _calculate_precision(self, mesh: trimesh.Trimesh, spheres: List[Sphere]) -> float:
        """
        Calculate precision: ratio of sphere volume that contains mesh.
        This measures how "tight" the spheres fit the mesh (minimizes empty space).
        
        Precision = (volume of mesh inside spheres) / (total sphere volume)
        
        Higher precision means less wasted space in spheres.
        """
        if len(spheres) == 0:
            return 0.0
        
        # Sample points inside sphere volumes
        n_samples_per_sphere = 100
        mesh_points_count = 0
        total_points_count = 0
        
        for sphere in spheres:
            # Sample random points inside this sphere
            for _ in range(n_samples_per_sphere):
                # Random point in sphere using rejection sampling
                while True:
                    # Random point in cube
                    offset = np.random.uniform(-sphere.radius, sphere.radius, 3)
                    if np.linalg.norm(offset) <= sphere.radius:
                        point = sphere.center + offset
                        break
                
                total_points_count += 1
                
                # Check if point is inside mesh
                if mesh.contains([point])[0]:
                    mesh_points_count += 1
        
        if total_points_count == 0:
            return 0.0
        
        precision = mesh_points_count / total_points_count
        return precision
    
    def _calculate_coverage(self, mesh: trimesh.Trimesh, 
                           spheres: List[Sphere]) -> float:
        """
        Calculate coverage ratio (backward compatibility).
        For detailed metrics, use _calculate_coverage_and_accuracy().
        """
        metrics = self._calculate_coverage_and_accuracy(mesh, spheres)
        return metrics['coverage_ratio']
    
    @staticmethod
    def compare_configs(mesh_path: str, 
                       configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare different configurations on the same mesh.
        
        Returns dict with metrics for each configuration.
        """
        results = []
        
        for i, config in enumerate(configs):
            converter = MeshToSpheresConverter(config)
            collection = converter.convert(mesh_path)
            
            results.append({
                'config_id': i,
                'config': config,
                'num_spheres': len(collection),
                'total_volume': collection.total_volume(),
                'coverage': collection.metadata['coverage_ratio'],
                'efficiency': collection.metadata['coverage_ratio'] / len(collection)
            })
        
        return results



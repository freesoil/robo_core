"""
Octree-based adaptive sphere generation.
Subdivides space until coverage and precision criteria are met.
"""

import numpy as np
from typing import List, Tuple
import trimesh
from dataclasses import dataclass


@dataclass
class OctreeNode:
    """Octree node representing a cubic region of space."""
    center: np.ndarray
    size: float
    contains_mesh: bool = False
    is_leaf: bool = True
    children: List['OctreeNode'] = None
    mesh_density: float = 0.0  # Fraction of cell containing mesh
    
    def get_sphere(self):
        """Convert this cubic cell to a sphere using circumsphere (diagonal radius)."""
        # Sphere circumscribed around cube (diagonal from center to corner)
        # This ensures the cube is completely contained in the sphere
        radius = self.size * np.sqrt(3) / 2.0
        return {'center': self.center, 'radius': radius}
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get min and max bounds of this cubic cell."""
        half_size = self.size / 2.0
        min_bound = self.center - half_size
        max_bound = self.center + half_size
        return min_bound, max_bound


class AdaptiveOctreeSpheres:
    """
    Adaptive octree-based sphere generation.
    
    Subdivides space hierarchically until quality criteria are met:
    - Coverage: % of mesh surface covered by spheres
    - Precision: % of sphere volume containing mesh (minimize empty space)
    """
    
    def __init__(self, mesh: trimesh.Trimesh, 
                 min_cell_size: float,
                 max_cell_size: float,
                 coverage_target: float = 0.95,
                 precision_target: float = 0.70,
                 min_radius: float = None):
        """
        Initialize octree sphere generator.
        
        Args:
            mesh: Input mesh
            min_cell_size: Minimum octree cell size (stopping criterion)
            max_cell_size: Initial octree cell size
            coverage_target: Target coverage ratio (0-1)
            precision_target: Target precision ratio (0-1)
            min_radius: Minimum sphere radius (for diagonal check)
        """
        self.mesh = mesh
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size
        self.coverage_target = coverage_target
        self.precision_target = precision_target
        self.min_radius = min_radius if min_radius is not None else min_cell_size / 2.0
        
        # Sample surface points for coverage calculation
        n_samples = max(1000, int(mesh.area * 500))
        self.surface_points, _ = trimesh.sample.sample_surface(mesh, n_samples)
        
        # Build initial octree
        self.root = self._build_initial_octree()
    
    def _build_initial_octree(self) -> OctreeNode:
        """Build initial octree root encompassing the mesh."""
        # Get mesh bounds
        bounds_min, bounds_max = self.mesh.bounds
        
        # Create cubic bounding box
        center = (bounds_min + bounds_max) / 2.0
        extents = bounds_max - bounds_min
        mesh_size = max(extents) * 1.1  # 10% padding
        
        # Choose initial size: use mesh size or max_cell_size, whichever is smaller
        # But ensure it's at least 4x min_cell_size to allow multiple subdivisions
        size = min(mesh_size, self.max_cell_size)
        
        # For very small meshes, ensure we can subdivide at least a few times
        # This prevents the "one big sphere" problem
        min_initial_size = self.min_cell_size * 8  # Allow at least 3 levels of subdivision
        if size < min_initial_size:
            # Very small mesh - use larger initial size to allow refinement
            size = min_initial_size
            # But don't exceed max_cell_size
            size = min(size, self.max_cell_size)
        
        root = OctreeNode(center=center, size=size)
        root.contains_mesh = True
        root.mesh_density = self._calculate_mesh_density(root)
        
        return root
    
    def _calculate_mesh_density(self, node: OctreeNode) -> float:
        """
        Calculate what fraction of this cubic cell contains mesh.
        
        Returns value between 0 and 1.
        """
        # Sample points in the cell
        n_samples = 50
        samples_in_mesh = 0
        
        min_bound, max_bound = node.get_bounds()
        
        for _ in range(n_samples):
            # Random point in cube
            point = np.random.uniform(min_bound, max_bound)
            
            # Check if in mesh
            if self.mesh.contains([point])[0]:
                samples_in_mesh += 1
        
        return samples_in_mesh / n_samples
    
    def _estimate_sphere_precision(self, node: OctreeNode) -> float:
        """
        Estimate the precision (mesh occupancy) of the sphere that would represent this cube.
        
        For a circumsphere:
        - Cube volume = size^3
        - Sphere volume = (4/3) * π * (radius)^3 where radius = size * √3 / 2
        - Mesh volume in cube ≈ cube_volume * mesh_density
        - Precision ≈ mesh_volume / sphere_volume
        
        Returns estimated precision (0-1)
        """
        if node.mesh_density < 0.01:
            return 0.0
        
        # Cube volume
        cube_volume = node.size ** 3
        
        # Circumsphere volume
        sphere_radius = node.size * np.sqrt(3) / 2.0
        sphere_volume = (4.0 / 3.0) * np.pi * (sphere_radius ** 3)
        
        # Estimated mesh volume in cube
        mesh_volume_in_cube = cube_volume * node.mesh_density
        
        # Precision = mesh_volume / sphere_volume
        # Note: mesh_volume_in_cube is an underestimate (only counts mesh in cube, not in sphere)
        # But it's a reasonable approximation
        precision = mesh_volume_in_cube / sphere_volume if sphere_volume > 0 else 0.0
        
        return precision
    
    def _should_subdivide(self, node: OctreeNode) -> bool:
        """
        Determine if this node should be subdivided.
        
        Subdivision strategy for good approximation:
        - Subdivide boundary regions (partial mesh occupancy)
        - Subdivide large cells that create low-precision spheres
        - Subdivide cells with low estimated precision (mostly empty spheres)
        - Stop at minimum size or when cell is mostly empty/full
        - Stop if cube diagonal would be < 2 * min_radius (sphere diameter)
        """
        # Don't subdivide if already at minimum size
        if node.size <= self.min_cell_size:
            return False
        
        # Don't subdivide if cube diagonal would be smaller than minimum sphere diameter
        cube_diagonal = node.size * np.sqrt(3)
        min_sphere_diameter = 2.0 * self.min_radius
        if cube_diagonal < min_sphere_diameter:
            return False
        
        # Don't subdivide if cell doesn't contain mesh
        if not node.contains_mesh or node.mesh_density < 0.01:
            return False
        
        # Estimate precision of sphere that would represent this cube
        estimated_precision = self._estimate_sphere_precision(node)
        
        # ALWAYS subdivide if estimated precision is below target
        # This catches large spheres with mostly empty space
        if estimated_precision < self.precision_target:
            return True
        
        # Subdivide large cells with any partial occupancy
        # Large cells with low density create low-precision spheres
        if node.size > self.min_cell_size * 2:
            # Subdivide if any partial occupancy (boundary regions)
            if 0.02 < node.mesh_density < 0.98:
                return True
        
        # Subdivide boundary regions (intermediate density)
        # These create spheres with mixed precision
        if 0.05 < node.mesh_density < 0.95:
            return True
        
        return False
    
    def _subdivide(self, node: OctreeNode):
        """Subdivide node into 8 children."""
        half_size = node.size / 2.0
        quarter_size = node.size / 4.0
        
        node.is_leaf = False
        node.children = []
        
        # Create 8 child nodes
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Calculate child center
                    offset = np.array([
                        (i - 0.5) * half_size,
                        (j - 0.5) * half_size,
                        (k - 0.5) * half_size
                    ])
                    child_center = node.center + offset
                    
                    child = OctreeNode(
                        center=child_center,
                        size=half_size
                    )
                    
                    # Check if child intersects mesh
                    min_bound, max_bound = child.get_bounds()
                    child.contains_mesh = self._cell_intersects_mesh(min_bound, max_bound)
                    
                    if child.contains_mesh:
                        child.mesh_density = self._calculate_mesh_density(child)
                    
                    node.children.append(child)
    
    def _cell_intersects_mesh(self, min_bound: np.ndarray, max_bound: np.ndarray) -> bool:
        """Check if cubic cell intersects with mesh."""
        # Check if any surface points are in this cell
        in_cell = np.all(
            (self.surface_points >= min_bound) & (self.surface_points <= max_bound),
            axis=1
        )
        
        if np.any(in_cell):
            return True
        
        # Also check if cell is inside mesh
        center = (min_bound + max_bound) / 2.0
        return self.mesh.contains([center])[0]
    
    def refine(self, max_iterations: int = 10):
        """
        Adaptively refine octree until criteria are met.
        
        Args:
            max_iterations: Maximum refinement iterations
        """
        for iteration in range(max_iterations):
            # Get current leaf nodes
            leaves = self._get_leaves(self.root)
            
            # Calculate current metrics
            coverage, precision = self._calculate_metrics(leaves)
            
            print(f"  Iteration {iteration + 1}: {len(leaves)} cells, "
                  f"Coverage={coverage:.1%}, Precision={precision:.1%}")
            
            # Check if criteria met
            if coverage >= self.coverage_target and precision >= self.precision_target:
                print(f"  ✓ Criteria met!")
                break
            
            # Find nodes to subdivide
            nodes_to_subdivide = [node for node in leaves if self._should_subdivide(node)]
            
            if not nodes_to_subdivide:
                print(f"  → Cannot subdivide further (min cell size reached)")
                # Check final metrics after stopping
                final_leaves = self._get_leaves(self.root)
                final_coverage, final_precision = self._calculate_metrics(final_leaves)
                print(f"  Final octree: {len(final_leaves)} cells, "
                      f"Coverage={final_coverage:.1%}, Precision={final_precision:.1%}")
                break
            
            # Subdivide nodes
            for node in nodes_to_subdivide:
                self._subdivide(node)
    
    def _find_optimal_sphere_for_region(self, points: np.ndarray, 
                                        min_radius: float, 
                                        max_radius: float) -> tuple:
        """
        Find optimal sphere (center and radius) to cover a region of points.
        Uses fine-grained radius search to find appropriately sized sphere.
        
        Args:
            points: Array of points in the region
            min_radius: Minimum allowed radius
            max_radius: Maximum allowed radius
        
        Returns:
            (best_center, best_radius, points_covered) or (None, None, 0) if no good fit
        """
        if len(points) == 0:
            return None, None, 0
        
        # Fine-grained radius search (more samples for better sizing)
        n_radius_samples = 40  # Even more fine-grained
        test_radii = np.linspace(max_radius, min_radius, n_radius_samples)
        
        best_center = None
        best_radius = max_radius
        best_coverage = 0
        best_score = -1.0
        
        # Try different centers: use points themselves, centroid, and some random samples
        center_candidates = []
        
        # 1. Centroid (often good for compact regions)
        center_candidates.append(np.mean(points, axis=0))
        
        # 2. Sample points as centers (good for distributed regions)
        n_point_samples = min(50, len(points)) # Increased from 20
        if len(points) > n_point_samples:
            indices = np.random.choice(len(points), n_point_samples, replace=False)
            center_candidates.extend(points[indices])
        else:
            center_candidates.extend(points)
        
        # 3. Try points near the boundary (for edge coverage)
        if len(points) > 5:
            # Find points near the convex hull
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                center_candidates.extend(hull_points[:min(20, len(hull_points))])
            except:
                pass  # If convex hull fails, skip
        
        for radius in test_radii:
            if radius < min_radius:
                continue
                
            # Pre-calculate distances for all candidates at once for efficiency
            for center in center_candidates:
                # Count points covered by this sphere
                dist_sq = np.sum((points - center)**2, axis=1)
                covered = np.sum(dist_sq <= radius**2)
                
                if covered == 0:
                    continue
                
                # Selection score: we want to maximize coverage but minimize radius
                # Using covered / (radius^2) strongly prefers smaller radii that cover points
                # We also add a small coverage weight to break ties towards more points
                score = covered / (radius**1.5) # Balanced metric
                
                if score > best_score:
                    best_score = score
                    best_coverage = covered
                    best_center = center.copy()
                    best_radius = radius
                elif abs(score - best_score) < 1e-6 and radius < best_radius:
                    # Tie-break with smaller radius
                    best_radius = radius
        
        # Only return if sphere covers at least 2 points and meets minimum radius
        if best_coverage >= 2 and best_radius >= min_radius:
            return best_center, best_radius, best_coverage
        
        return None, None, 0
        
        # Only return if sphere covers at least 2 points and meets minimum radius
        if best_coverage >= 2 and best_radius >= min_radius:
            return best_center, best_radius, best_coverage
        
        return None, None, 0
    
    def _add_coverage_spheres(self, existing_spheres: List) -> List:
        """
        Add additional spheres to improve coverage after octree refinement.
        These spheres are not constrained to octree cells and can be placed optimally.
        
        Uses fine-grained sphere sizing to find appropriately sized spheres for each region.
        Allows multiple spheres to cover a region for better precision.
        
        Args:
            existing_spheres: List of Sphere objects from octree
            
        Returns:
            List of additional Sphere objects
        """
        from .converter import Sphere
        
        # Find uncovered surface points
        uncovered_points = []
        for point in self.surface_points:
            if not any(sphere.contains_point(point) for sphere in existing_spheres):
                uncovered_points.append(point)
        
        if len(uncovered_points) == 0:
            return []  # Already fully covered
        
        uncovered_points = np.array(uncovered_points)
        print(f"  Coverage stage: {len(uncovered_points)} uncovered surface points")
        
        # Place spheres to cover uncovered points
        # Use fine-grained greedy algorithm with appropriate sphere sizing
        additional_spheres = []
        remaining_points = uncovered_points.copy()
        max_sphere_radius = self.max_cell_size / 2.0  # Max radius from max cell size
        min_sphere_radius = self.min_radius  # Must respect minimum
        
        max_additional_spheres = 300  # Allow more spheres for fine-grained coverage
        min_points_per_sphere = 2  # Minimum points to cover (low threshold for fine coverage)
        
        iteration = 0
        
        while len(remaining_points) > 0 and iteration < max_additional_spheres:
            iteration += 1
            
            if len(remaining_points) == 0:
                break
            
            # Determine appropriate max radius for this region
            # Use local point distribution to size sphere appropriately
            if len(remaining_points) > 1:
                # Calculate spread of remaining points
                centroid = np.mean(remaining_points, axis=0)
                distances_to_centroid = np.linalg.norm(remaining_points - centroid, axis=1)
                local_max_distance = np.percentile(distances_to_centroid, 60)  # Lowered from 75th
                
                # Use smaller max radius for compact regions, larger for spread out
                # Tighter multiplier (1.2 instead of 1.5)
                region_max_radius = min(max_sphere_radius, max(local_max_distance * 1.2, min_sphere_radius * 1.5))
            else:
                region_max_radius = min_sphere_radius * 2
            
            # Find optimal sphere for remaining points with fine-grained search
            best_center, best_radius, points_covered = self._find_optimal_sphere_for_region(
                remaining_points,
                min_radius=min_sphere_radius,
                max_radius=region_max_radius
            )
            
            # If no good sphere found with local max, try with global max
            if best_center is None and region_max_radius < max_sphere_radius:
                best_center, best_radius, points_covered = self._find_optimal_sphere_for_region(
                    remaining_points,
                    min_radius=min_sphere_radius,
                    max_radius=max_sphere_radius
                )
            
            # Only add sphere if it covers enough points and meets minimum radius
            if best_center is None or points_covered < min_points_per_sphere or best_radius < min_sphere_radius:
                break
            
            # Ensure radius meets minimum requirement
            best_radius = max(best_radius, min_sphere_radius)
            
            # Create sphere
            best_sphere = Sphere(center=best_center, radius=best_radius)
            
            # Add best sphere
            additional_spheres.append(best_sphere)
            
            # Remove covered points
            distances = np.linalg.norm(remaining_points - best_sphere.center, axis=1)
            remaining_points = remaining_points[distances > best_sphere.radius]
            
            if len(remaining_points) == 0:
                break
        
        if len(additional_spheres) > 0:
            # Calculate statistics
            radii = [s.radius for s in additional_spheres]
            print(f"  Added {len(additional_spheres)} coverage spheres "
                  f"(radius range: {min(radii):.4f}-{max(radii):.4f}, "
                  f"avg: {np.mean(radii):.4f}), "
                  f"{len(remaining_points)} points still uncovered")
        else:
            print(f"  No additional spheres needed or couldn't place effective spheres")
        
        return additional_spheres
    
    def _get_leaves(self, node: OctreeNode) -> List[OctreeNode]:
        """Get all leaf nodes in subtree."""
        if node.is_leaf:
            return [node]
        
        leaves = []
        if node.children:
            for child in node.children:
                leaves.extend(self._get_leaves(child))
        
        return leaves
    
    def _calculate_metrics(self, leaves: List[OctreeNode]) -> Tuple[float, float]:
        """
        Calculate coverage and precision from leaf nodes using SPHERES (circumspheres).
        
        Coverage: % of surface points inside spheres
        Precision: % of sphere volume containing mesh
        """
        from .converter import Sphere
        
        # Convert nodes to spheres
        spheres = []
        for node in leaves:
            if node.contains_mesh and node.mesh_density > 0.01:
                sphere_dict = node.get_sphere()  # Uses circumsphere (diagonal radius)
                spheres.append(Sphere(
                    center=sphere_dict['center'],
                    radius=sphere_dict['radius']
                ))
        
        if len(spheres) == 0:
            return 0.0, 0.0
        
        # Coverage: % of surface points inside any sphere
        covered_points = 0
        for point in self.surface_points:
            if any(sphere.contains_point(point) for sphere in spheres):
                covered_points += 1
        
        coverage = covered_points / len(self.surface_points) if len(self.surface_points) > 0 else 0.0
        
        # Precision: % of sphere volume containing mesh
        # Sample points inside each sphere and check mesh occupancy
        total_samples = 0
        mesh_samples = 0
        
        for sphere in spheres:
            # Sample points inside sphere
            n_samples = 100
            for _ in range(n_samples):
                # Random point in sphere using rejection sampling
                while True:
                    offset = np.random.uniform(-sphere.radius, sphere.radius, 3)
                    if np.linalg.norm(offset) <= sphere.radius:
                        point = sphere.center + offset
                        break
                
                total_samples += 1
                if self.mesh.contains([point])[0]:
                    mesh_samples += 1
        
        precision = mesh_samples / total_samples if total_samples > 0 else 0.0
        
        return coverage, precision
    
    def get_spheres(self, add_coverage_spheres: bool = True) -> List[dict]:
        """
        Get spheres representing the octree leaves, optionally adding coverage spheres.
        Uses circumspheres (diagonal radius) to ensure cubes are fully contained.
        
        Args:
            add_coverage_spheres: If True, add additional spheres to meet coverage target
        
        Returns:
            List of dicts with 'center' and 'radius'
        """
        from .converter import Sphere
        
        leaves = self._get_leaves(self.root)
        
        # Get octree spheres
        octree_spheres = []
        for node in leaves:
            if node.contains_mesh and node.mesh_density > 0.01:
                # Use circumsphere (diagonal radius) - ensures cube is fully contained
                sphere_dict = node.get_sphere()  # Now returns circumsphere
                octree_spheres.append(Sphere(
                    center=sphere_dict['center'],
                    radius=sphere_dict['radius']
                ))
        
        # Check if we need additional coverage spheres
        if add_coverage_spheres:
            # Calculate current coverage
            coverage, _ = self._calculate_metrics(leaves)
            
            if coverage < self.coverage_target:
                print(f"  Coverage ({coverage:.1%}) below target ({self.coverage_target:.1%})")
                print(f"  Adding coverage spheres to improve coverage...")
                
                # Add spheres to cover uncovered areas
                additional_spheres = self._add_coverage_spheres(octree_spheres)
                
                # Combine octree and coverage spheres
                all_spheres = octree_spheres + additional_spheres
                
                # Verify final coverage
                final_coverage, final_precision = self._calculate_metrics_from_spheres(all_spheres)
                print(f"  Final: {len(all_spheres)} total spheres, "
                      f"Coverage={final_coverage:.1%}, Precision={final_precision:.1%}")
                
                # Convert to dict format
                return [{'center': s.center, 'radius': s.radius} for s in all_spheres]
        
        # Return only octree spheres
        return [{'center': s.center, 'radius': s.radius} for s in octree_spheres]
    
    def _calculate_metrics_from_spheres(self, spheres: List) -> Tuple[float, float]:
        """
        Calculate coverage and precision from a list of Sphere objects.
        """
        if len(spheres) == 0:
            return 0.0, 0.0
        
        # Coverage: % of surface points inside any sphere
        covered_points = 0
        for point in self.surface_points:
            if any(sphere.contains_point(point) for sphere in spheres):
                covered_points += 1
        
        coverage = covered_points / len(self.surface_points) if len(self.surface_points) > 0 else 0.0
        
        # Precision: % of sphere volume containing mesh
        total_samples = 0
        mesh_samples = 0
        
        for sphere in spheres:
            # Sample points inside sphere
            n_samples = 100
            for _ in range(n_samples):
                # Random point in sphere using rejection sampling
                while True:
                    offset = np.random.uniform(-sphere.radius, sphere.radius, 3)
                    if np.linalg.norm(offset) <= sphere.radius:
                        point = sphere.center + offset
                        break
                
                total_samples += 1
                if self.mesh.contains([point])[0]:
                    mesh_samples += 1
        
        precision = mesh_samples / total_samples if total_samples > 0 else 0.0
        
        return coverage, precision


def octree_adaptive_spheres(mesh: trimesh.Trimesh,
                           min_radius: float,
                           max_radius: float,
                           coverage_target: float = 0.95,
                           precision_target: float = 0.70) -> List:
    """
    Generate spheres using adaptive octree subdivision.
    
    Args:
        mesh: Input mesh
        min_radius: Minimum sphere radius (min cell size)
        max_radius: Maximum sphere radius (initial cell size)
        coverage_target: Target coverage ratio
        precision_target: Target precision ratio
        
    Returns:
        List of Sphere objects
    """
    from .converter import Sphere
    
    # Cell size = 2 * radius (diameter)
    min_cell_size = min_radius * 2.0
    max_cell_size = max_radius * 2.0
    
    print(f"  Building adaptive octree...")
    print(f"  Cell size range: {min_cell_size:.4f} to {max_cell_size:.4f}")
    print(f"  Min sphere radius: {min_radius:.4f} (stops when cube diagonal < {2*min_radius:.4f})")
    print(f"  Target coverage: {coverage_target:.1%}, precision: {precision_target:.1%}")
    
    # Build and refine octree
    octree = AdaptiveOctreeSpheres(
        mesh,
        min_cell_size=min_cell_size,
        max_cell_size=max_cell_size,
        coverage_target=coverage_target,
        precision_target=precision_target,
        min_radius=min_radius
    )
    
    octree.refine(max_iterations=15)
    
    # Convert to spheres
    sphere_dicts = octree.get_spheres()
    spheres = [Sphere(center=s['center'], radius=s['radius']) for s in sphere_dicts]
    
    print(f"  Generated {len(spheres)} spheres")
    
    return spheres


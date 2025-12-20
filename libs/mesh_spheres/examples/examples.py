"""
Example usage scripts for mesh_spheres library.
"""

import numpy as np
from pathlib import Path


def example_basic_conversion():
    """Basic mesh to spheres conversion."""
    from mesh_spheres import MeshToSpheresConverter
    
    print("=== Basic Conversion Example ===\n")
    
    # Use default configuration
    converter = MeshToSpheresConverter()
    
    # Convert a mesh
    mesh_path = 'path/to/your/mesh.stl'
    collection = converter.convert(mesh_path)
    
    print(f"Generated {len(collection)} spheres")
    print(f"Coverage: {collection.metadata['coverage_ratio']:.2%}")
    print(f"Total volume: {collection.total_volume():.6f}")
    
    # Access individual spheres
    print("\nFirst 3 spheres:")
    for i, sphere in enumerate(collection.spheres[:3]):
        print(f"  {i+1}. Center: {sphere.center}, Radius: {sphere.radius:.4f}")
    
    return collection


def example_compare_methods():
    """Compare different conversion methods."""
    from mesh_spheres import MeshToSpheresConverter
    
    print("\n=== Comparing Methods ===\n")
    
    mesh_path = 'path/to/your/mesh.stl'
    
    configs = [
        {'method': 'adaptive', 'target_spheres': 80},
        {'method': 'hierarchical', 'target_spheres': 30},
        {'method': 'medial_axis', 'sample_density': 5000},
    ]
    
    results = MeshToSpheresConverter.compare_configs(mesh_path, configs)
    
    print(f"{'Method':<20} {'Spheres':<10} {'Coverage':<12} {'Efficiency':<12}")
    print("-" * 54)
    
    for result in results:
        method = result['config']['method']
        spheres = result['num_spheres']
        coverage = result['coverage']
        efficiency = result['efficiency']
        
        print(f"{method:<20} {spheres:<10} {coverage:<12.2%} {efficiency:<12.4f}")
    
    return results


def example_collision_detection():
    """Demonstrate collision detection capabilities."""
    from mesh_spheres import MeshToSpheresConverter, Sphere
    
    print("\n=== Collision Detection Example ===\n")
    
    # Convert mesh
    mesh_path = 'path/to/your/mesh.stl'
    converter = MeshToSpheresConverter()
    collection = converter.convert(mesh_path)
    
    # Test point collision
    test_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.1, 0.1]),
        np.array([1.0, 1.0, 1.0]),
    ]
    
    print("Point collision tests:")
    for i, point in enumerate(test_points):
        is_inside = collection.contains_point(point)
        distance = collection.distance_to_point(point)
        print(f"  Point {i+1} {point}: Inside={is_inside}, Distance={distance:.4f}")
    
    # Test sphere-sphere collision
    test_sphere = Sphere(center=np.array([0.05, 0.05, 0.05]), radius=0.02)
    
    colliding_spheres = [
        s for s in collection.spheres 
        if test_sphere.intersects_sphere(s)
    ]
    
    print(f"\nTest sphere collides with {len(colliding_spheres)} spheres")
    
    return collection


def example_2d_projection():
    """Demonstrate 3D to 2D projection."""
    from mesh_spheres import MeshToSpheresConverter
    
    print("\n=== 2D Projection Example ===\n")
    
    # Convert mesh
    mesh_path = 'path/to/your/mesh.stl'
    converter = MeshToSpheresConverter()
    collection = converter.convert(mesh_path)
    
    # Define camera parameters
    # Simple pinhole camera model
    focal_length = 800
    cx, cy = 320, 240  # Image center
    
    camera_matrix = np.array([
        [focal_length, 0, cx, 0],
        [0, focal_length, cy, 0],
        [0, 0, 1, 0]
    ], dtype=float)
    
    image_size = (640, 480)
    
    # Project spheres
    projections = collection.project_to_2d(camera_matrix, image_size)
    
    print(f"Projected {len(projections)}/{len(collection)} spheres")
    print("\nFirst 3 projections:")
    for i, (center_2d, radius_2d) in enumerate(projections[:3]):
        print(f"  {i+1}. Center: ({center_2d[0]:.1f}, {center_2d[1]:.1f}), "
              f"Radius: {radius_2d:.1f} pixels")
    
    return projections


def example_visualization():
    """Demonstrate visualization capabilities."""
    from mesh_spheres import MeshToSpheresConverter
    from mesh_spheres.visualization import (
        visualize_spheres,
        compare_methods,
        plot_coverage_vs_spheres
    )
    import trimesh
    import matplotlib.pyplot as plt
    
    print("\n=== Visualization Example ===\n")
    
    mesh_path = 'path/to/your/mesh.stl'
    mesh = trimesh.load(mesh_path)
    
    # Convert mesh
    converter = MeshToSpheresConverter()
    collection = converter.convert(mesh_path)
    
    # Visualize spheres with mesh overlay
    print("Displaying sphere visualization...")
    fig1, ax1 = visualize_spheres(collection, mesh=mesh)
    plt.show()
    
    # Compare methods side-by-side
    print("Comparing methods...")
    fig2 = compare_methods(mesh_path)
    plt.show()
    
    # Plot coverage vs number of spheres
    print("Plotting coverage curve...")
    fig3 = plot_coverage_vs_spheres(mesh_path, method='adaptive', max_spheres=150)
    plt.show()
    
    print("Visualization complete!")


def example_robot_arm_conversion():
    """Convert all links of a robot arm."""
    from mesh_spheres import MeshToSpheresConverter
    
    print("\n=== Robot Arm Conversion Example ===\n")
    
    # Path to robot meshes
    robot_mesh_dir = Path('lerobot_curobot_sim/isaac_assets/SO_ARM100/meshes')
    
    if not robot_mesh_dir.exists():
        print(f"Directory not found: {robot_mesh_dir}")
        print("Update the path to match your setup.")
        return
    
    # Create converter with balanced config
    config = {
        'method': 'adaptive',
        'target_spheres': 60,
        'min_radius_ratio': 0.015,
        'max_radius_ratio': 0.25,
        'coverage_threshold': 0.93,
    }
    converter = MeshToSpheresConverter(config)
    
    # Convert all STL files
    sphere_collections = {}
    output_dir = Path('sphere_models')
    output_dir.mkdir(exist_ok=True)
    
    print(f"{'Link Name':<30} {'Spheres':<10} {'Coverage':<12} {'Volume':<15}")
    print("-" * 67)
    
    for mesh_file in sorted(robot_mesh_dir.glob('*.stl')):
        link_name = mesh_file.stem
        
        try:
            collection = converter.convert(str(mesh_file))
            sphere_collections[link_name] = collection
            
            # Print stats
            num_spheres = len(collection)
            coverage = collection.metadata['coverage_ratio']
            volume = collection.total_volume()
            
            print(f"{link_name:<30} {num_spheres:<10} {coverage:<12.2%} {volume:<15.6f}")
            
            # Save collection
            output_file = output_dir / f"{link_name}_spheres.npz"
            collection.save(str(output_file))
            
        except Exception as e:
            print(f"{link_name:<30} ERROR: {e}")
    
    print(f"\nSaved {len(sphere_collections)} sphere collections to {output_dir}")
    
    return sphere_collections


def example_custom_configuration():
    """Example of customizing configuration for specific needs."""
    from mesh_spheres import MeshToSpheresConverter
    
    print("\n=== Custom Configuration Examples ===\n")
    
    mesh_path = 'path/to/your/mesh.stl'
    
    # High accuracy configuration
    print("1. High Accuracy Configuration:")
    high_accuracy_config = {
        'method': 'adaptive',
        'target_spheres': 150,
        'min_radius_ratio': 0.01,
        'max_radius_ratio': 0.2,
        'coverage_threshold': 0.98,
    }
    converter1 = MeshToSpheresConverter(high_accuracy_config)
    collection1 = converter1.convert(mesh_path)
    print(f"   Spheres: {len(collection1)}, Coverage: {collection1.metadata['coverage_ratio']:.2%}")
    
    # Fast collision detection configuration
    print("\n2. Fast Collision Detection Configuration:")
    fast_config = {
        'method': 'hierarchical',
        'target_spheres': 25,
        'min_radius_ratio': 0.025,
        'max_radius_ratio': 0.35,
    }
    converter2 = MeshToSpheresConverter(fast_config)
    collection2 = converter2.convert(mesh_path)
    print(f"   Spheres: {len(collection2)}, Coverage: {collection2.metadata['coverage_ratio']:.2%}")
    
    # Skeleton-based configuration
    print("\n3. Skeleton-Based Configuration:")
    skeleton_config = {
        'method': 'medial_axis',
        'min_radius_ratio': 0.02,
        'max_radius_ratio': 0.25,
        'sample_density': 8000,
    }
    converter3 = MeshToSpheresConverter(skeleton_config)
    collection3 = converter3.convert(mesh_path)
    print(f"   Spheres: {len(collection3)}, Coverage: {collection3.metadata['coverage_ratio']:.2%}")


def example_save_and_load():
    """Demonstrate saving and loading sphere collections."""
    from mesh_spheres import MeshToSpheresConverter, SphereCollection
    
    print("\n=== Save and Load Example ===\n")
    
    mesh_path = 'path/to/your/mesh.stl'
    save_path = 'sphere_collection.npz'
    
    # Convert and save
    print("Converting mesh...")
    converter = MeshToSpheresConverter()
    collection = converter.convert(mesh_path)
    
    print(f"Saving to {save_path}...")
    collection.save(save_path)
    
    # Load
    print(f"Loading from {save_path}...")
    loaded_collection = SphereCollection.load(save_path)
    
    print(f"Loaded {len(loaded_collection)} spheres")
    print(f"Metadata: {loaded_collection.metadata}")
    
    # Verify they match
    assert len(collection) == len(loaded_collection)
    print("✓ Save/load successful!")
    
    return loaded_collection


if __name__ == '__main__':
    print("Mesh to Spheres Library - Examples")
    print("=" * 50)
    
    # Run examples (uncomment the ones you want to try)
    
    # example_basic_conversion()
    # example_compare_methods()
    # example_collision_detection()
    # example_2d_projection()
    # example_visualization()
    # example_robot_arm_conversion()
    # example_custom_configuration()
    # example_save_and_load()
    
    print("\n" + "=" * 50)
    print("Examples complete!")
    print("\nTo run specific examples, uncomment them in the __main__ block.")





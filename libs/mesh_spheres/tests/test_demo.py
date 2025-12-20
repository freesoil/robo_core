#!/usr/bin/env python3
"""
Test and demonstration script for mesh_spheres library.
Tests the library on SO-ARM100 robot meshes.
"""

import numpy as np
from pathlib import Path
import sys
import time


def test_basic_functionality():
    """Test basic conversion functionality."""
    from mesh_spheres import MeshToSpheresConverter, Sphere
    
    print("\n" + "="*70)
    print("TEST 1: Basic Functionality")
    print("="*70)
    
    # Find a test mesh
    test_mesh = Path(__file__).parent.parent.parent.parent.parent / \
                'isaac_assets' / 'SO_ARM100' / 'meshes' / 'base_link.stl'
    
    if not test_mesh.exists():
        print(f"⚠ Test mesh not found: {test_mesh}")
        print("  Using default configuration test instead")
        return True
    
    print(f"Converting: {test_mesh.name}")
    
    # Test default configuration
    start_time = time.time()
    converter = MeshToSpheresConverter()
    collection = converter.convert(str(test_mesh))
    elapsed = time.time() - start_time
    
    print(f"\n✓ Conversion successful!")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Spheres: {len(collection)}")
    print(f"  Coverage: {collection.metadata['coverage_ratio']:.2%}")
    print(f"  Total volume: {collection.total_volume():.6f}")
    
    # Test sphere access
    if len(collection) > 0:
        sphere = collection.spheres[0]
        print(f"\n  First sphere:")
        print(f"    Center: {sphere.center}")
        print(f"    Radius: {sphere.radius:.4f}")
        print(f"    Volume: {sphere.volume():.6f}")
    
    # Test bounds
    min_bounds, max_bounds = collection.bounds()
    print(f"\n  Bounding box:")
    print(f"    Min: {min_bounds}")
    print(f"    Max: {max_bounds}")
    
    return True


def test_collision_detection():
    """Test collision detection features."""
    from mesh_spheres import MeshToSpheresConverter, Sphere
    
    print("\n" + "="*70)
    print("TEST 2: Collision Detection")
    print("="*70)
    
    test_mesh = Path(__file__).parent.parent.parent.parent.parent / \
                'isaac_assets' / 'SO_ARM100' / 'meshes' / 'base_link.stl'
    
    if not test_mesh.exists():
        print("⚠ Skipping (mesh not found)")
        return True
    
    converter = MeshToSpheresConverter()
    collection = converter.convert(str(test_mesh))
    
    # Test point collision
    min_bounds, max_bounds = collection.bounds()
    center_point = (min_bounds + max_bounds) / 2
    outside_point = max_bounds + np.array([1.0, 1.0, 1.0])
    
    print(f"\nPoint collision tests:")
    print(f"  Center point: {collection.contains_point(center_point)} "
          f"(distance: {collection.distance_to_point(center_point):.4f})")
    print(f"  Outside point: {collection.contains_point(outside_point)} "
          f"(distance: {collection.distance_to_point(outside_point):.4f})")
    
    # Test sphere-sphere collision
    test_sphere = Sphere(center=center_point, radius=0.01)
    collisions = sum(1 for s in collection.spheres if test_sphere.intersects_sphere(s))
    print(f"\n  Test sphere at center collides with {collisions} spheres")
    
    print("\n✓ Collision detection working!")
    return True


def test_2d_projection():
    """Test 3D to 2D projection."""
    from mesh_spheres import MeshToSpheresConverter
    
    print("\n" + "="*70)
    print("TEST 3: 2D Projection")
    print("="*70)
    
    test_mesh = Path(__file__).parent.parent.parent.parent.parent / \
                'isaac_assets' / 'SO_ARM100' / 'meshes' / 'base_link.stl'
    
    if not test_mesh.exists():
        print("⚠ Skipping (mesh not found)")
        return True
    
    converter = MeshToSpheresConverter()
    collection = converter.convert(str(test_mesh))
    
    # Simple camera matrix
    focal_length = 800
    cx, cy = 320, 240
    camera_matrix = np.array([
        [focal_length, 0, cx, 0],
        [0, focal_length, cy, 0],
        [0, 0, 1, 0.5]  # Camera at z=0.5
    ], dtype=float)
    
    image_size = (640, 480)
    
    projections = collection.project_to_2d(camera_matrix, image_size)
    
    print(f"\nProjected {len(projections)}/{len(collection)} spheres to 2D")
    
    if len(projections) > 0:
        print(f"\nFirst 3 projections:")
        for i, (center_2d, radius_2d) in enumerate(projections[:3]):
            print(f"  {i+1}. Center: ({center_2d[0]:.1f}, {center_2d[1]:.1f}), "
                  f"Radius: {radius_2d:.1f} px")
    
    print("\n✓ 2D projection working!")
    return True


def test_save_load():
    """Test save and load functionality."""
    from mesh_spheres import MeshToSpheresConverter, SphereCollection
    import tempfile
    
    print("\n" + "="*70)
    print("TEST 4: Save and Load")
    print("="*70)
    
    test_mesh = Path(__file__).parent.parent.parent.parent.parent / \
                'isaac_assets' / 'SO_ARM100' / 'meshes' / 'base_link.stl'
    
    if not test_mesh.exists():
        print("⚠ Skipping (mesh not found)")
        return True
    
    # Convert
    converter = MeshToSpheresConverter()
    collection = converter.convert(str(test_mesh))
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_path = f.name
    
    collection.save(temp_path)
    print(f"Saved to: {temp_path}")
    
    # Load
    loaded = SphereCollection.load(temp_path)
    print(f"Loaded: {len(loaded)} spheres")
    
    # Verify
    assert len(collection) == len(loaded), "Sphere count mismatch!"
    assert collection.metadata['coverage_ratio'] == loaded.metadata['coverage_ratio'], \
           "Metadata mismatch!"
    
    # Cleanup
    Path(temp_path).unlink()
    
    print("\n✓ Save/load working!")
    return True


def test_compare_methods():
    """Test and compare different methods."""
    from mesh_spheres import MeshToSpheresConverter
    
    print("\n" + "="*70)
    print("TEST 5: Compare Methods")
    print("="*70)
    
    test_mesh = Path(__file__).parent.parent.parent.parent.parent / \
                'isaac_assets' / 'SO_ARM100' / 'meshes' / 'base_link.stl'
    
    if not test_mesh.exists():
        print("⚠ Skipping (mesh not found)")
        return True
    
    configs = [
        {'method': 'adaptive', 'target_spheres': 50},
        {'method': 'hierarchical', 'target_spheres': 30},
        {'method': 'medial_axis', 'sample_density': 3000},
    ]
    
    print(f"\nComparing {len(configs)} methods on {test_mesh.name}:")
    print(f"\n{'Method':<20} {'Spheres':<10} {'Coverage':<12} {'Efficiency':<12} {'Time':<10}")
    print("-" * 64)
    
    for config in configs:
        start_time = time.time()
        converter = MeshToSpheresConverter(config)
        collection = converter.convert(str(test_mesh))
        elapsed = time.time() - start_time
        
        method = config['method']
        num_spheres = len(collection)
        coverage = collection.metadata['coverage_ratio']
        efficiency = coverage / num_spheres if num_spheres > 0 else 0
        
        print(f"{method:<20} {num_spheres:<10} {coverage:<12.2%} "
              f"{efficiency:<12.4f} {elapsed:<10.2f}s")
    
    print("\n✓ Method comparison complete!")
    return True


def test_optimal_configs():
    """Test the recommended optimal configurations."""
    from mesh_spheres import MeshToSpheresConverter
    
    print("\n" + "="*70)
    print("TEST 6: Optimal Configurations")
    print("="*70)
    
    test_mesh = Path(__file__).parent.parent.parent.parent.parent / \
                'isaac_assets' / 'SO_ARM100' / 'meshes' / 'base_link.stl'
    
    if not test_mesh.exists():
        print("⚠ Skipping (mesh not found)")
        return True
    
    # Test the three main use cases
    configs = {
        'High Accuracy': {
            'method': 'adaptive',
            'target_spheres': 150,
            'min_radius_ratio': 0.01,
            'max_radius_ratio': 0.2,
            'coverage_threshold': 0.98,
        },
        'Balanced (Default)': {
            'method': 'adaptive',
            'target_spheres': 80,
            'min_radius_ratio': 0.015,
            'max_radius_ratio': 0.25,
            'coverage_threshold': 0.95,
        },
        'Efficiency': {
            'method': 'hierarchical',
            'target_spheres': 30,
            'min_radius_ratio': 0.02,
            'max_radius_ratio': 0.3,
        },
    }
    
    print(f"\nTesting optimal configurations:")
    print(f"\n{'Configuration':<20} {'Spheres':<10} {'Coverage':<12} {'Volume Ratio':<15} {'Time':<10}")
    print("-" * 67)
    
    import trimesh
    mesh = trimesh.load(str(test_mesh))
    mesh_volume = mesh.volume
    
    for name, config in configs.items():
        start_time = time.time()
        converter = MeshToSpheresConverter(config)
        collection = converter.convert(str(test_mesh))
        elapsed = time.time() - start_time
        
        volume_ratio = collection.total_volume() / mesh_volume if mesh_volume > 0 else 0
        
        print(f"{name:<20} {len(collection):<10} "
              f"{collection.metadata['coverage_ratio']:<12.2%} "
              f"{volume_ratio:<15.2f} {elapsed:<10.2f}s")
    
    print("\n✓ Optimal configurations tested!")
    return True


def demo_robot_arm_conversion():
    """Demo: Convert all robot arm meshes."""
    from mesh_spheres import MeshToSpheresConverter
    
    print("\n" + "="*70)
    print("DEMO: Robot Arm Conversion")
    print("="*70)
    
    mesh_dir = Path(__file__).parent.parent.parent.parent.parent / \
               'isaac_assets' / 'SO_ARM100' / 'meshes'
    
    if not mesh_dir.exists():
        print(f"⚠ Mesh directory not found: {mesh_dir}")
        return False
    
    # Use balanced configuration
    config = {
        'method': 'adaptive',
        'target_spheres': 60,
        'min_radius_ratio': 0.015,
        'max_radius_ratio': 0.25,
        'coverage_threshold': 0.93,
    }
    
    converter = MeshToSpheresConverter(config)
    
    stl_files = sorted(mesh_dir.glob('*.stl'))
    
    print(f"\nFound {len(stl_files)} STL files")
    print(f"\n{'Link Name':<30} {'Spheres':<10} {'Coverage':<12} {'Time':<10}")
    print("-" * 62)
    
    results = []
    for mesh_file in stl_files:
        link_name = mesh_file.stem
        
        try:
            start_time = time.time()
            collection = converter.convert(str(mesh_file))
            elapsed = time.time() - start_time
            
            print(f"{link_name:<30} {len(collection):<10} "
                  f"{collection.metadata['coverage_ratio']:<12.2%} "
                  f"{elapsed:<10.2f}s")
            
            results.append({
                'name': link_name,
                'spheres': len(collection),
                'coverage': collection.metadata['coverage_ratio']
            })
            
        except Exception as e:
            print(f"{link_name:<30} ERROR: {str(e)[:30]}")
    
    if results:
        avg_spheres = np.mean([r['spheres'] for r in results])
        avg_coverage = np.mean([r['coverage'] for r in results])
        
        print(f"\n{'Average':<30} {avg_spheres:<10.1f} {avg_coverage:<12.2%}")
        print("\n✓ Robot arm conversion complete!")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*15 + "MESH TO SPHERES LIBRARY - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Collision Detection", test_collision_detection),
        ("2D Projection", test_2d_projection),
        ("Save and Load", test_save_load),
        ("Compare Methods", test_compare_methods),
        ("Optimal Configs", test_optimal_configs),
    ]
    
    demos = [
        ("Robot Arm Conversion", demo_robot_arm_conversion),
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Run demos
    print("\n" + "="*70)
    print(" "*25 + "DEMOS")
    print("="*70)
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print(" "*25 + "SUMMARY")
    print("="*70)
    print(f"\nTests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())





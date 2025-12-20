#!/usr/bin/env python3
"""
Convert SO-ARM100 robot meshes to sphere collections.

This script converts all STL meshes of the SO-ARM100 robot to sphere collections
and saves them for use in collision detection and projection tasks.
"""

import numpy as np
from pathlib import Path
import sys
import time
import json


def convert_so100_robot(output_dir='sphere_models', visualize=False):
    """
    Convert all SO-ARM100 robot links to sphere collections.
    
    Args:
        output_dir: Directory to save sphere collections
        visualize: Whether to generate visualizations
    """
    from mesh_spheres import MeshToSpheresConverter
    
    # Find mesh directory
    script_dir = Path(__file__).parent
    mesh_dir = script_dir.parent.parent.parent.parent.parent / \
               'isaac_assets' / 'SO_ARM100' / 'meshes'
    
    if not mesh_dir.exists():
        print(f"Error: Mesh directory not found: {mesh_dir}")
        print("Please update the path in the script.")
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*70)
    print(" "*15 + "SO-ARM100 ROBOT MESH CONVERSION")
    print("="*70)
    print(f"\nMesh directory: {mesh_dir}")
    print(f"Output directory: {output_path.absolute()}")
    
    # Use optimized configuration for robot links
    # Balanced for accuracy and efficiency
    config = {
        'method': 'adaptive',
        'target_spheres': 70,  # Good for typical robot links
        'min_radius_ratio': 0.015,
        'max_radius_ratio': 0.25,
        'coverage_threshold': 0.94,
        'sample_density': 5000,
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    converter = MeshToSpheresConverter(config)
    
    # Find all STL files
    stl_files = sorted(mesh_dir.glob('*.stl'))
    
    if not stl_files:
        print(f"\nNo STL files found in {mesh_dir}")
        return False
    
    print(f"\nFound {len(stl_files)} STL files")
    print("\nConverting...")
    print("-"*70)
    print(f"{'Link Name':<30} {'Spheres':<10} {'Coverage':<12} {'Time':<10}")
    print("-"*70)
    
    results = []
    total_time = 0
    
    for mesh_file in stl_files:
        link_name = mesh_file.stem
        
        try:
            start_time = time.time()
            collection = converter.convert(str(mesh_file))
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Save collection
            output_file = output_path / f"{link_name}_spheres.npz"
            collection.save(str(output_file))
            
            # Store results
            result = {
                'link_name': link_name,
                'mesh_file': str(mesh_file),
                'num_spheres': len(collection),
                'coverage': collection.metadata['coverage_ratio'],
                'total_volume': collection.total_volume(),
                'bounds': [
                    collection.bounds()[0].tolist(),
                    collection.bounds()[1].tolist()
                ],
                'output_file': str(output_file),
                'conversion_time': elapsed,
            }
            results.append(result)
            
            # Print progress
            print(f"{link_name:<30} {len(collection):<10} "
                  f"{collection.metadata['coverage_ratio']:<12.2%} "
                  f"{elapsed:<10.2f}s")
            
            # Optional visualization
            if visualize:
                try:
                    from mesh_spheres.visualization import visualize_sphere_collection as visualize_spheres
                    import trimesh
                    import matplotlib.pyplot as plt
                    
                    mesh = trimesh.load(str(mesh_file))
                    fig, ax = visualize_spheres(collection, mesh=mesh)
                    
                    vis_file = output_path / f"{link_name}_visualization.png"
                    plt.savefig(str(vis_file), dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    result['visualization'] = str(vis_file)
                except Exception as e:
                    print(f"  Warning: Could not create visualization: {e}")
            
        except Exception as e:
            print(f"{link_name:<30} ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary statistics
    print("-"*70)
    
    if results:
        avg_spheres = np.mean([r['num_spheres'] for r in results])
        avg_coverage = np.mean([r['coverage'] for r in results])
        total_spheres = sum(r['num_spheres'] for r in results)
        
        print(f"{'Summary':<30} {avg_spheres:<10.1f} {avg_coverage:<12.2%} "
              f"{total_time:<10.2f}s")
        
        print("\n" + "="*70)
        print("CONVERSION COMPLETE")
        print("="*70)
        print(f"\nTotal links: {len(results)}")
        print(f"Total spheres: {total_spheres}")
        print(f"Average spheres per link: {avg_spheres:.1f}")
        print(f"Average coverage: {avg_coverage:.1%}")
        print(f"Total conversion time: {total_time:.2f}s")
        print(f"Average time per link: {total_time/len(results):.2f}s")
        
        # Save summary JSON
        summary_file = output_path / 'conversion_summary.json'
        summary = {
            'robot': 'SO-ARM100',
            'conversion_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': config,
            'total_links': len(results),
            'total_spheres': total_spheres,
            'average_spheres': avg_spheres,
            'average_coverage': avg_coverage,
            'total_time': total_time,
            'links': results,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
        print(f"Sphere collections saved to: {output_path.absolute()}")
        
        # Usage instructions
        print("\n" + "="*70)
        print("USAGE INSTRUCTIONS")
        print("="*70)
        print("\nTo load a sphere collection:")
        print("```python")
        print("from mesh_spheres import SphereCollection")
        print(f"collection = SphereCollection.load('{output_path}/base_link_spheres.npz')")
        print("```")
        print("\nTo use in collision detection:")
        print("```python")
        print("import numpy as np")
        print("point = np.array([0.1, 0.2, 0.3])")
        print("is_collision = collection.contains_point(point)")
        print("```")
        print("\nSee README.md for more usage examples.")
        
        return True
    
    return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert SO-ARM100 robot meshes to sphere collections'
    )
    parser.add_argument(
        '--output-dir',
        default='sphere_models',
        help='Output directory for sphere collections (default: sphere_models)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization images (requires matplotlib)'
    )
    
    args = parser.parse_args()
    
    success = convert_so100_robot(
        output_dir=args.output_dir,
        visualize=args.visualize
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())





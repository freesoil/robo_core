#!/usr/bin/env python3
"""
Convert all meshes from an XML (URDF/MuJoCo) file to sphere collections.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

import sys
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
libs_dir = script_dir.parent.parent  # robo_core/libs
if str(libs_dir) not in sys.path:
    sys.path.insert(0, str(libs_dir))

from mesh_spheres.xml_parser import RobotMeshParser


def convert_xml_to_spheres(xml_file: str, 
                          output_dir: str = 'sphere_models',
                          config: Dict = None) -> Dict:
    """
    Convert all STL meshes referenced in XML to sphere collections.
    
    Args:
        xml_file: Path to URDF or MuJoCo XML file
        output_dir: Directory to save sphere collections
        config: Optional MeshToSpheresConverter configuration
        
    Returns:
        Results dictionary with conversion metadata
    """
    from mesh_spheres import MeshToSpheresConverter
    
    print("="*70)
    print(" "*15 + "XML TO SPHERES CONVERTER")
    print("="*70)
    
    # Parse XML to extract STL files
    print(f"\nParsing XML file: {xml_file}")
    parser = RobotMeshParser(xml_file)
    parser.print_summary()
    
    stl_files = parser.get_unique_stl_files()
    
    if not stl_files:
        print("\n⚠ No STL files found in XML!")
        return {'error': 'No STL files found'}
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Use default config if none provided
    if config is None:
        config = {
            'min_radius_ratio': 0.012,
            'max_radius_ratio': 0.30,
            'coverage_threshold': 0.95,
            'precision_threshold': 0.70,
        }
    
    print(f"\nConverter configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create converter
    converter = MeshToSpheresConverter(config)
    
    # Convert all STL files
    print(f"\n{'='*90}")
    print("CONVERTING MESHES")
    print(f"{'='*90}\n")
    print(f"{'Mesh Name':<28} {'Spheres':<10} {'Coverage':<11} {'Precision':<11} {'RMS Err':<11} {'Time':<8}")
    print("-"*90)
    
    results = []
    total_time = 0
    
    for stl_file in stl_files:
        mesh_name = Path(stl_file).stem
        
        # Don't catch exceptions - let them propagate for debugging
        start_time = time.time()
        collection = converter.convert(stl_file)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Save collection
        output_file = output_path / f"{mesh_name}_spheres.npz"
        collection.save(str(output_file))
        
        # Store results
        result = {
            'mesh_name': mesh_name,
            'stl_file': stl_file,
            'num_spheres': len(collection),
            'coverage': collection.metadata['coverage_ratio'],
            'precision': collection.metadata.get('precision_ratio', 0.0),
            'mean_error': collection.metadata.get('mean_error', 0.0),
            'rms_error': collection.metadata.get('rms_error', 0.0),
            'max_error': collection.metadata.get('max_error', 0.0),
            'efficiency': collection.metadata.get('efficiency', 0.0),
            'total_volume': collection.total_volume(),
            'bounds': [
                collection.bounds()[0].tolist(),
                collection.bounds()[1].tolist()
            ],
            'output_file': str(output_file),
            'conversion_time': elapsed,
        }
        results.append(result)
        
        # Print progress with new metrics
        precision = collection.metadata.get('precision_ratio', 0.0)
        rms_error = collection.metadata.get('rms_error', 0.0)
        print(f"{mesh_name:<28} {len(collection):<10} "
              f"{collection.metadata['coverage_ratio']:<11.1%} "
              f"{precision:<11.1%} "
              f"{rms_error:<11.4f} "
              f"{elapsed:<8.2f}s")
    
    # Summary
    print("-"*90)
    
    successful_results = [r for r in results if 'error' not in r]
    
    if successful_results:
        avg_spheres = np.mean([r['num_spheres'] for r in successful_results])
        avg_coverage = np.mean([r['coverage'] for r in successful_results])
        avg_precision = np.mean([r['precision'] for r in successful_results])
        avg_rms_error = np.mean([r['rms_error'] for r in successful_results])
        total_spheres = sum(r['num_spheres'] for r in successful_results)
        
        print(f"{'Summary':<28} {avg_spheres:<10.1f} {avg_coverage:<11.1%} "
              f"{avg_precision:<11.1%} "
              f"{avg_rms_error:<11.4f} "
              f"{total_time:<8.2f}s")
        
        print("\n" + "="*70)
        print("CONVERSION COMPLETE")
        print("="*70)
        print(f"\nTotal meshes: {len(stl_files)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(results) - len(successful_results)}")
        print(f"Total spheres: {total_spheres}")
        print(f"Average spheres per mesh: {avg_spheres:.1f}")
        print(f"Average coverage: {avg_coverage:.1%} (target: {config['coverage_threshold']:.1%})")
        print(f"Average precision: {avg_precision:.1%} (target: {config['precision_threshold']:.1%})")
        print(f"Average RMS error: {avg_rms_error:.4f}")
        print(f"Total conversion time: {total_time:.2f}s")
        print(f"\nQuality Metrics:")
        print(f"  Coverage (surface covered): {avg_coverage:.1%}")
        print(f"  Precision (mesh occupancy): {avg_precision:.1%}")
        print(f"  Mean error: {np.mean([r['mean_error'] for r in successful_results]):.4f}m")
        print(f"  RMS error: {avg_rms_error:.4f}m")
        print(f"  Max error: {np.max([r['max_error'] for r in successful_results]):.4f}m")
        print(f"  Efficiency (coverage/sphere): {np.mean([r['efficiency'] for r in successful_results]):.6f}")
        
        # Save summary JSON
        summary_file = output_path / 'conversion_summary.json'
        summary = {
            'xml_file': str(xml_file),
            'xml_format': parser.format,
            'conversion_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': config,
            'total_meshes': len(stl_files),
            'successful_conversions': len(successful_results),
            'total_spheres': total_spheres,
            'average_spheres': float(avg_spheres),
            'average_coverage': float(avg_coverage),
            'total_time': total_time,
            'meshes': results,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
        print(f"Sphere collections saved to: {output_path.absolute()}")
        
        return summary
    else:
        print("\n⚠ No successful conversions!")
        return {'error': 'All conversions failed', 'results': results}


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert all meshes from XML (URDF/MuJoCo) to sphere collections'
    )
    parser.add_argument(
        'xml_file',
        help='Path to URDF or MuJoCo XML file'
    )
    parser.add_argument(
        '--output-dir',
        default='sphere_models',
        help='Output directory for sphere collections (default: sphere_models)'
    )
    # Method removed - only octree_adaptive is used now
    parser.add_argument(
        '--min-radius',
        type=float,
        default=None,
        help='Minimum sphere radius in meters (overrides min-radius-ratio)'
    )
    parser.add_argument(
        '--max-radius',
        type=float,
        default=None,
        help='Maximum sphere radius in meters (overrides max-radius-ratio)'
    )
    parser.add_argument(
        '--min-radius-ratio',
        type=float,
        default=0.015,
        help='Min radius as fraction of mesh size (default: 0.015)'
    )
    parser.add_argument(
        '--max-radius-ratio',
        type=float,
        default=0.25,
        help='Max radius as fraction of mesh size (default: 0.25)'
    )
    parser.add_argument(
        '--target-coverage',
        type=float,
        default=0.95,
        help='Target coverage ratio 0-1 (default: 0.95 = 95%% surface coverage)'
    )
    parser.add_argument(
        '--target-precision',
        type=float,
        default=0.70,
        help='Target precision ratio 0-1 (default: 0.70 = 70%% mesh occupancy, minimizes empty space)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Launch interactive visualizer after conversion'
    )
    
    args = parser.parse_args()
    
    # Build config (octree method is always used)
    config = {
        'min_radius_ratio': args.min_radius_ratio,
        'max_radius_ratio': args.max_radius_ratio,
        'coverage_threshold': args.target_coverage,
        'precision_threshold': args.target_precision,
    }
    
    # If absolute radius values provided, add them to config
    if args.min_radius is not None:
        config['min_radius_absolute'] = args.min_radius
    if args.max_radius is not None:
        config['max_radius_absolute'] = args.max_radius
    
    # Convert
    summary = convert_xml_to_spheres(
        args.xml_file,
        output_dir=args.output_dir,
        config=config
    )
    
    # Launch visualizer if requested
    if args.visualize and 'error' not in summary:
        print("\n" + "="*70)
        print("Launching interactive visualizer...")
        print("="*70)
        
        try:
            from visualizer_open3d import launch_visualizer
            launch_visualizer(args.xml_file, args.output_dir)
        except ImportError as e:
            print(f"⚠ Could not launch visualizer: {e}")
            print("Make sure open3d is installed: pip install open3d")
    
    return 0 if 'error' not in summary else 1


if __name__ == '__main__':
    sys.exit(main())


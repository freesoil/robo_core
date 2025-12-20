#!/usr/bin/env python3
"""
MuJoCo XML generator for visualizing mesh and sphere representations.
Creates a MuJoCo XML file with:
- Group 0: Floor
- Group 1: Original meshes
- Group 2: Sphere representations
- Joint controls
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Handle imports for both script and module usage
try:
    from ..converter import SphereCollection
except ImportError:
    # If running as script, add libs directory to path
    script_dir = Path(__file__).parent
    libs_dir = script_dir.parent  # robo_core/libs
    if str(libs_dir) not in sys.path:
        sys.path.insert(0, str(libs_dir))
    from mesh_spheres import SphereCollection


class MuJoCoVisualizer:
    """Generate MuJoCo XML for visualizing mesh and sphere representations."""
    
    def __init__(self, xml_file: str, sphere_dir: str):
        """
        Initialize visualizer.
        
        Args:
            xml_file: Path to original URDF/MuJoCo XML file
            sphere_dir: Directory containing sphere .npz files
        """
        self.xml_file = Path(xml_file)
        self.sphere_dir = Path(sphere_dir)
        self.xml_tree = None
        self.sphere_collections = {}
        
        # Load XML
        self._load_xml()
        
        # Load sphere collections
        self._load_spheres()
    
    def _load_xml(self):
        """Load and parse the XML file."""
        self.xml_tree = ET.parse(self.xml_file)
        self.xml_root = self.xml_tree.getroot()
    
    def _load_spheres(self):
        """Load all sphere collections from sphere_dir."""
        if not self.sphere_dir.exists():
            raise ValueError(f"Sphere directory not found: {self.sphere_dir}")
        
        print(f"\nLoading sphere collections from: {self.sphere_dir}")
        # Find all .npz files
        for npz_file in self.sphere_dir.glob("*.npz"):
            if "_metadata.pkl" in str(npz_file):
                continue
            
            mesh_name = npz_file.stem.replace("_spheres", "")
            try:
                collection = SphereCollection.load(str(npz_file))
                self.sphere_collections[mesh_name] = collection
                print(f"  ✓ Loaded {len(collection)} spheres for '{mesh_name}'")
            except Exception as e:
                print(f"  ✗ Warning: Could not load {npz_file}: {e}")
        
        print(f"\nAvailable sphere collections: {list(self.sphere_collections.keys())}")
    
    def _extract_joints_from_urdf(self) -> List[Dict]:
        """Extract joint information from URDF."""
        joints = []
        
        # Find all joint elements
        for joint in self.xml_root.findall('.//joint'):
            joint_info = {
                'name': joint.get('name', ''),
                'type': joint.get('type', 'revolute'),
                'parent': None,
                'child': None,
                'axis': [0, 0, 1],  # Default axis
                'origin': {'xyz': [0, 0, 0], 'rpy': [0, 0, 0]},
                'limit': {'lower': -3.14, 'upper': 3.14, 'effort': 100, 'velocity': 10}
            }
            
            # Get parent and child links
            parent_elem = joint.find('parent')
            child_elem = joint.find('child')
            if parent_elem is not None:
                joint_info['parent'] = parent_elem.get('link', '')
            if child_elem is not None:
                joint_info['child'] = child_elem.get('link', '')
            
            # Get axis
            axis_elem = joint.find('axis')
            if axis_elem is not None:
                xyz = axis_elem.get('xyz', '0 0 1')
                joint_info['axis'] = [float(x) for x in xyz.split()]
            
            # Get origin
            origin_elem = joint.find('origin')
            if origin_elem is not None:
                xyz = origin_elem.get('xyz', '0 0 0')
                rpy = origin_elem.get('rpy', '0 0 0')
                joint_info['origin']['xyz'] = [float(x) for x in xyz.split()]
                joint_info['origin']['rpy'] = [float(x) for x in rpy.split()]
            
            # Get limits
            limit_elem = joint.find('limit')
            if limit_elem is not None:
                joint_info['limit']['lower'] = float(limit_elem.get('lower', -3.14))
                joint_info['limit']['upper'] = float(limit_elem.get('upper', 3.14))
                joint_info['limit']['effort'] = float(limit_elem.get('effort', 100))
                joint_info['limit']['velocity'] = float(limit_elem.get('velocity', 10))
            
            joints.append(joint_info)
        
        return joints
    
    def _extract_joints_from_mujoco(self) -> List[Dict]:
        """Extract joint information from MuJoCo XML."""
        joints = []
        
        # Find all joint elements
        for joint in self.xml_root.findall('.//joint'):
            joint_info = {
                'name': joint.get('name', ''),
                'type': joint.get('type', 'hinge'),
                'axis': [0, 0, 1],
                'pos': [0, 0, 0],
                'range': [-3.14, 3.14],
                'damping': 0.1
            }
            
            # Get axis
            axis = joint.get('axis', '0 0 1')
            joint_info['axis'] = [float(x) for x in axis.split()]
            
            # Get position
            pos = joint.get('pos', '0 0 0')
            joint_info['pos'] = [float(x) for x in pos.split()]
            
            # Get range
            range_str = joint.get('range', '-3.14 3.14')
            joint_info['range'] = [float(x) for x in range_str.split()]
            
            # Get damping
            joint_info['damping'] = float(joint.get('damping', '0.1'))
            
            joints.append(joint_info)
        
        return joints
    
    def _extract_mesh_paths(self) -> Dict[str, str]:
        """Extract mesh file paths from XML."""
        mesh_paths = {}
        
        # Try URDF format first
        for link in self.xml_root.findall('.//link'):
            link_name = link.get('name', 'unknown')
            
            # Find visual meshes in this link
            for visual in link.findall('.//visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is not None:
                    filename = mesh.get('filename', '')
                    if filename:
                        mesh_paths[link_name] = filename
        
        # Try MuJoCo format
        asset = self.xml_root.find('asset')
        if asset is not None:
            for mesh in asset.findall('mesh'):
                name = mesh.get('name', '')
                file_attr = mesh.get('file', '')
                if name and file_attr:
                    mesh_paths[name] = file_attr
        
        return mesh_paths
    
    def _copy_element_attributes(self, src: ET.Element, dst: ET.Element):
        """Copy all attributes from source to destination element."""
        for key, value in src.attrib.items():
            dst.set(key, value)
    
    def _find_matching_sphere_collection(self, mesh_name: str) -> Optional[str]:
        """Find matching sphere collection for a mesh name, handling name variations."""
        # Direct match
        if mesh_name in self.sphere_collections:
            return mesh_name
        
        # Try case-insensitive match
        mesh_name_lower = mesh_name.lower()
        for key in self.sphere_collections.keys():
            if key.lower() == mesh_name_lower:
                return key
        
        # Try removing common suffixes/prefixes
        variations = [
            mesh_name.replace('_Collision_1', '').replace('_Collision_2', '').replace('_Collision_3', ''),
            mesh_name.replace('_visual', '').replace('_Visual', ''),
            mesh_name.replace('_collision', '').replace('_Collision', ''),
        ]
        for variant in variations:
            if variant in self.sphere_collections:
                return variant
            # Case-insensitive variant match
            variant_lower = variant.lower()
            for key in self.sphere_collections.keys():
                if key.lower() == variant_lower:
                    return key
        
        return None
    
    def _add_spheres_to_body(self, body_elem: ET.Element, mesh_name: str, geom_counter: int = 0):
        """Add sphere geoms to a body for a given mesh name."""
        if mesh_name in self.sphere_collections:
            collection = self.sphere_collections[mesh_name]
            body_name = body_elem.get('name', 'unnamed')
            # Make names unique by including body name and geom counter
            base_name = f'sphere_{body_name}_{mesh_name}_{geom_counter}'
            # Sanitize name (MuJoCo doesn't like some characters)
            base_name = base_name.replace(' ', '_').replace('-', '_')
            
            for i, sphere in enumerate(collection.spheres):
                sphere_geom = ET.SubElement(body_elem, 'geom',
                                           name=f'{base_name}_{i}',
                                           type='sphere',
                                           size=f'{sphere.radius:.6f}',
                                           pos=f'{sphere.center[0]:.6f} {sphere.center[1]:.6f} {sphere.center[2]:.6f}',
                                           group='2',
                                           rgba='1.0 0.3 0.3 0.7',  # Red, semi-transparent
                                           contype='0', conaffinity='0')  # No collision
            print(f"    Added {len(collection.spheres)} spheres for mesh '{mesh_name}' in body '{body_name}'")
    
    def _copy_body_recursive(self, src_body: ET.Element, dst_parent: ET.Element, mesh_base_path: Path):
        """Recursively copy body structure and add sphere representations."""
        # Create new body element
        dst_body = ET.SubElement(dst_parent, 'body')
        self._copy_element_attributes(src_body, dst_body)
        
        # Track which meshes have been added to this body to avoid duplicates
        meshes_added = set()
        geom_counter = 0
        
        # Copy all child elements except bodies (we'll handle those recursively)
        for child in src_body:
            if child.tag == 'body':
                # Recursively copy child bodies
                self._copy_body_recursive(child, dst_body, mesh_base_path)
            elif child.tag == 'geom':
                # Copy geom and check if it uses a mesh
                geom = ET.SubElement(dst_body, 'geom')
                self._copy_element_attributes(child, geom)
                
                # If this geom uses a mesh, add original mesh (Group 1) and spheres (Group 2)
                mesh_attr = geom.get('mesh', '')
                if mesh_attr:
                    # ALWAYS force group to 1 for original mesh (override any existing group)
                    geom.set('group', '1')
                    geom.set('rgba', '0.7 0.7 0.9 0.6')  # Blue, semi-transparent
                    geom.set('contype', '0')
                    geom.set('conaffinity', '0')
                    
                    # Mesh name is the mesh attribute value directly
                    mesh_name = mesh_attr
                    
                    # Try to find matching sphere collection (handle name variations)
                    sphere_mesh_name = self._find_matching_sphere_collection(mesh_name)
                    
                    # Add sphere representations (Group 2) only once per mesh per body
                    if sphere_mesh_name and sphere_mesh_name not in meshes_added:
                        self._add_spheres_to_body(dst_body, sphere_mesh_name, geom_counter)
                        meshes_added.add(sphere_mesh_name)
                    elif not sphere_mesh_name:
                        print(f"  Warning: No sphere collection found for mesh '{mesh_name}'")
                    geom_counter += 1
            else:
                # Copy other elements as-is (inertial, joint, etc.)
                new_elem = ET.SubElement(dst_body, child.tag)
                self._copy_element_attributes(child, new_elem)
                # Copy text content if any
                if child.text:
                    new_elem.text = child.text
                if child.tail:
                    new_elem.tail = child.tail
    
    def _create_mujoco_xml(self, output_file: str):
        """Create MuJoCo XML file preserving original structure with sphere representations."""
        # Determine if input is URDF or MuJoCo
        is_urdf = self.xml_root.tag == 'robot' or 'urdf' in str(self.xml_file).lower()
        
        if is_urdf:
            raise NotImplementedError("URDF support not yet implemented. Please use MuJoCo XML format.")
        
        # For MuJoCo XML, preserve the entire structure
        # Create MuJoCo XML root
        mujoco = ET.Element('mujoco')
        self._copy_element_attributes(self.xml_root, mujoco)
        if not mujoco.get('model'):
            mujoco.set('model', 'mesh_sphere_visualization')
        
        # Copy compiler, option, default sections
        for section in ['compiler', 'option', 'default']:
            src_section = self.xml_root.find(section)
            if src_section is not None:
                dst_section = ET.SubElement(mujoco, section)
                self._copy_element_attributes(src_section, dst_section)
                # Recursively copy nested elements
                for child in src_section:
                    self._copy_nested_element(child, dst_section)
        
        # Copy and enhance asset section
        src_asset = self.xml_root.find('asset')
        if src_asset is not None:
            asset = ET.SubElement(mujoco, 'asset')
            # Copy all asset children
            for child in src_asset:
                new_child = ET.SubElement(asset, child.tag)
                self._copy_element_attributes(child, new_child)
                if child.text:
                    new_child.text = child.text
            
            # Add floor texture and material
            floor_texture = ET.SubElement(asset, 'texture', name='floor', type='2d', 
                                         builtin='checker', width='100', height='100', 
                                         rgb1='0.2 0.3 0.4', rgb2='0.1 0.2 0.3')
            floor_material = ET.SubElement(asset, 'material', name='floor', texture='floor', texrepeat='1 1')
        else:
            asset = ET.SubElement(mujoco, 'asset')
            floor_texture = ET.SubElement(asset, 'texture', name='floor', type='2d', 
                                         builtin='checker', width='100', height='100', 
                                         rgb1='0.2 0.3 0.4', rgb2='0.1 0.2 0.3')
            floor_material = ET.SubElement(asset, 'material', name='floor', texture='floor', texrepeat='1 1')
        
        # Copy worldbody and add floor + sphere representations
        src_worldbody = self.xml_root.find('worldbody')
        if src_worldbody is not None:
            worldbody = ET.SubElement(mujoco, 'worldbody')
            
            # Add floor (Group 0)
            floor = ET.SubElement(worldbody, 'geom', name='floor', type='plane', 
                                 size='10 10 0.1', rgba='0.5 0.5 0.5 1', 
                                 group='0', material='floor')
            
            # Copy all bodies recursively, adding sphere representations
            mesh_base_path = self.xml_file.parent
            for src_body in src_worldbody.findall('body'):
                self._copy_body_recursive(src_body, worldbody, mesh_base_path)
            
            # Copy other worldbody children (geoms, lights, etc.)
            for child in src_worldbody:
                if child.tag == 'geom':
                    # Copy geom and handle mesh geoms
                    geom = ET.SubElement(worldbody, 'geom')
                    self._copy_element_attributes(child, geom)
                    
                    # If this geom uses a mesh, force it to group 1
                    mesh_attr = geom.get('mesh', '')
                    if mesh_attr:
                        geom.set('group', '1')
                        geom.set('rgba', '0.7 0.7 0.9 0.6')
                        geom.set('contype', '0')
                        geom.set('conaffinity', '0')
                        
                        # Try to add spheres for this mesh
                        mesh_name = mesh_attr
                        sphere_mesh_name = self._find_matching_sphere_collection(mesh_name)
                        if sphere_mesh_name:
                            self._add_spheres_to_body(worldbody, sphere_mesh_name, 0)
                        else:
                            print(f"  Warning: No sphere collection found for worldbody mesh '{mesh_name}'")
                elif child.tag != 'body':
                    new_child = ET.SubElement(worldbody, child.tag)
                    self._copy_element_attributes(child, new_child)
        
        # Copy actuator section
        src_actuator = self.xml_root.find('actuator')
        if src_actuator is not None:
            actuator = ET.SubElement(mujoco, 'actuator')
            for child in src_actuator:
                new_child = ET.SubElement(actuator, child.tag)
                self._copy_element_attributes(child, new_child)
                if child.text:
                    new_child.text = child.text
        
        # Copy other top-level sections (sensor, contact, keyframe, etc.)
        for section in ['sensor', 'contact', 'keyframe', 'include']:
            src_section = self.xml_root.find(section)
            if src_section is not None:
                dst_section = ET.SubElement(mujoco, section)
                self._copy_element_attributes(src_section, dst_section)
                for child in src_section:
                    self._copy_nested_element(child, dst_section)
        
        # Write XML to file
        tree = ET.ElementTree(mujoco)
        ET.indent(tree, space='  ')
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        # Count statistics
        num_spheres = sum(len(c.spheres) for c in self.sphere_collections.values())
        num_meshes = len([g for g in mujoco.findall('.//geom[@mesh]')])
        
        print(f"\n✓ Created MuJoCo XML: {output_file}")
        print(f"  - Floor (Group 0): 1 geom")
        print(f"  - Original meshes (Group 1): {num_meshes} geoms")
        print(f"  - Sphere representations (Group 2): {num_spheres} geoms")
        print(f"  - Structure preserved from original XML")
    
    def _copy_nested_element(self, src: ET.Element, dst_parent: ET.Element):
        """Recursively copy nested XML elements."""
        new_elem = ET.SubElement(dst_parent, src.tag)
        self._copy_element_attributes(src, new_elem)
        if src.text:
            new_elem.text = src.text
        if src.tail:
            new_elem.tail = src.tail
        for child in src:
            self._copy_nested_element(child, new_elem)
    
    def generate(self, output_file: str):
        """
        Generate MuJoCo XML file.
        
        Args:
            output_file: Path to output MuJoCo XML file
        """
        self._create_mujoco_xml(output_file)


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate MuJoCo XML for visualizing mesh and sphere representations'
    )
    parser.add_argument('xml_file', help='Input URDF/MuJoCo XML file')
    parser.add_argument('--sphere-dir', required=True,
                       help='Directory containing sphere .npz files')
    parser.add_argument('--output', '-o', default='visualization.xml',
                       help='Output MuJoCo XML file (default: visualization.xml)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MuJoCo Visualization Generator")
    print("=" * 70)
    print(f"Input XML: {args.xml_file}")
    print(f"Sphere directory: {args.sphere_dir}")
    print(f"Output: {args.output}")
    print()
    
    try:
        visualizer = MuJoCoVisualizer(args.xml_file, args.sphere_dir)
        visualizer.generate(args.output)
        
        print("\n" + "=" * 70)
        print("Usage:")
        print("=" * 70)
        print(f"  # View in MuJoCo viewer")
        print(f"  mujoco-viewer {args.output}")
        print()
        print(f"  # Or in Python:")
        print(f"  import mujoco")
        print(f"  model = mujoco.MjModel.from_xml_path('{args.output}')")
        print(f"  viewer = mujoco.MjViewer(model)")
        print()
        print("Controls:")
        print("  - Toggle groups: Press 0, 1, 2 to show/hide floor, meshes, spheres")
        print("  - Control joints: Use keyboard or Python API")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


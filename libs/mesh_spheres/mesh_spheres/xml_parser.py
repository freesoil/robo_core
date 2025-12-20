"""
XML parser for extracting mesh files from URDF and MuJoCo XML files.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple
import re


class RobotMeshParser:
    """Parse URDF and MuJoCo XML files to extract mesh references."""
    
    def __init__(self, xml_file: str):
        """
        Initialize parser with XML file.
        
        Args:
            xml_file: Path to URDF or MuJoCo XML file
        """
        self.xml_file = Path(xml_file)
        self.xml_dir = self.xml_file.parent
        
        if not self.xml_file.exists():
            raise FileNotFoundError(f"XML file not found: {xml_file}")
        
        # Parse XML
        self.tree = ET.parse(self.xml_file)
        self.root = self.tree.getroot()
        
        # Detect format
        self.format = self._detect_format()
        
    def _detect_format(self) -> str:
        """Detect if XML is URDF or MuJoCo format."""
        if self.root.tag == 'robot':
            return 'urdf'
        elif self.root.tag == 'mujoco':
            return 'mujoco'
        else:
            raise ValueError(f"Unknown XML format. Root tag: {self.root.tag}")
    
    def extract_meshes(self) -> List[Dict[str, str]]:
        """
        Extract all mesh file references from XML.
        
        Returns:
            List of dicts with keys: 'name', 'file', 'type', 'link'
        """
        if self.format == 'urdf':
            return self._extract_urdf_meshes()
        elif self.format == 'mujoco':
            return self._extract_mujoco_meshes()
    
    def _extract_urdf_meshes(self) -> List[Dict[str, str]]:
        """Extract meshes from URDF format."""
        meshes = []
        
        # Find all links
        for link in self.root.findall('.//link'):
            link_name = link.get('name', 'unknown')
            
            # Find visual meshes
            for visual in link.findall('.//visual'):
                visual_name = visual.get('name', '')
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh_elem = geometry.find('mesh')
                    if mesh_elem is not None:
                        filename = mesh_elem.get('filename', '')
                        if filename:
                            meshes.append({
                                'name': visual_name or f"{link_name}_visual",
                                'file': filename,
                                'type': 'visual',
                                'link': link_name,
                                'format': 'urdf'
                            })
            
            # Find collision meshes
            for collision in link.findall('.//collision'):
                geometry = collision.find('geometry')
                if geometry is not None:
                    mesh_elem = geometry.find('mesh')
                    if mesh_elem is not None:
                        filename = mesh_elem.get('filename', '')
                        if filename:
                            meshes.append({
                                'name': f"{link_name}_collision",
                                'file': filename,
                                'type': 'collision',
                                'link': link_name,
                                'format': 'urdf'
                            })
        
        return meshes
    
    def _extract_mujoco_meshes(self) -> List[Dict[str, str]]:
        """Extract meshes from MuJoCo format."""
        meshes = []
        
        # Get mesh directory from compiler
        meshdir = ''
        compiler = self.root.find('compiler')
        if compiler is not None:
            meshdir = compiler.get('meshdir', '')
        
        # Find all mesh assets
        asset = self.root.find('asset')
        if asset is not None:
            for mesh in asset.findall('mesh'):
                mesh_name = mesh.get('name', 'unknown')
                mesh_file = mesh.get('file', '')
                
                if mesh_file:
                    # Combine meshdir with file
                    if meshdir:
                        full_path = str(Path(meshdir) / mesh_file)
                    else:
                        full_path = mesh_file
                    
                    meshes.append({
                        'name': mesh_name,
                        'file': full_path,
                        'type': 'asset',
                        'link': '',  # MuJoCo doesn't have explicit links in asset
                        'format': 'mujoco'
                    })
        
        return meshes
    
    def resolve_mesh_paths(self, meshes: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Resolve relative mesh paths to absolute paths.
        
        Args:
            meshes: List of mesh dicts from extract_meshes()
            
        Returns:
            Updated mesh list with 'absolute_path' key
        """
        resolved = []
        
        for mesh in meshes:
            mesh_copy = mesh.copy()
            relative_path = mesh['file']
            
            # Try to resolve path
            # First try relative to XML directory
            abs_path = self.xml_dir / relative_path
            
            if abs_path.exists():
                mesh_copy['absolute_path'] = str(abs_path)
                mesh_copy['exists'] = True
            else:
                # Try without prefix (e.g., "meshes/Base.stl" -> just look for Base.stl nearby)
                filename = Path(relative_path).name
                search_path = self.xml_dir / filename
                
                if search_path.exists():
                    mesh_copy['absolute_path'] = str(search_path)
                    mesh_copy['exists'] = True
                else:
                    # Try looking in meshes subdirectory
                    meshes_dir = self.xml_dir / 'meshes'
                    if meshes_dir.exists():
                        mesh_path = meshes_dir / filename
                        if mesh_path.exists():
                            mesh_copy['absolute_path'] = str(mesh_path)
                            mesh_copy['exists'] = True
                        else:
                            mesh_copy['absolute_path'] = str(abs_path)
                            mesh_copy['exists'] = False
                    else:
                        mesh_copy['absolute_path'] = str(abs_path)
                        mesh_copy['exists'] = False
            
            resolved.append(mesh_copy)
        
        return resolved
    
    def get_unique_stl_files(self) -> List[str]:
        """
        Get unique list of STL files (filters out duplicates and non-STL).
        
        Returns:
            List of absolute paths to unique STL files
        """
        meshes = self.extract_meshes()
        resolved = self.resolve_mesh_paths(meshes)
        
        # Filter for STL files that exist
        stl_files = set()
        for mesh in resolved:
            if mesh['exists']:
                file_path = Path(mesh['absolute_path'])
                # Check if it's an STL file
                if file_path.suffix.lower() in ['.stl']:
                    stl_files.add(str(file_path))
        
        return sorted(list(stl_files))
    
    def get_mesh_info(self) -> List[Dict[str, str]]:
        """
        Get detailed information about all meshes.
        
        Returns:
            List of mesh info dicts with all metadata
        """
        meshes = self.extract_meshes()
        resolved = self.resolve_mesh_paths(meshes)
        
        # Filter for existing files
        return [m for m in resolved if m['exists']]
    
    def print_summary(self):
        """Print summary of meshes found in XML."""
        meshes = self.extract_meshes()
        resolved = self.resolve_mesh_paths(meshes)
        
        print(f"\n{'='*70}")
        print(f"XML MESH PARSER SUMMARY")
        print(f"{'='*70}")
        print(f"File: {self.xml_file}")
        print(f"Format: {self.format.upper()}")
        print(f"Total meshes found: {len(meshes)}")
        
        # Count by type
        visual_count = sum(1 for m in meshes if m['type'] == 'visual')
        collision_count = sum(1 for m in meshes if m['type'] == 'collision')
        asset_count = sum(1 for m in meshes if m['type'] == 'asset')
        
        if visual_count:
            print(f"  Visual meshes: {visual_count}")
        if collision_count:
            print(f"  Collision meshes: {collision_count}")
        if asset_count:
            print(f"  Asset meshes: {asset_count}")
        
        # Count existing files
        existing = sum(1 for m in resolved if m['exists'])
        missing = len(resolved) - existing
        
        print(f"\nFile status:")
        print(f"  Found: {existing}")
        print(f"  Missing: {missing}")
        
        # Get unique STL files
        stl_files = self.get_unique_stl_files()
        print(f"\nUnique STL files: {len(stl_files)}")
        
        if stl_files:
            print(f"\n{'Filename':<40} {'Size':<10} {'Status':<10}")
            print("-"*60)
            for stl_file in stl_files:
                path = Path(stl_file)
                size = path.stat().st_size if path.exists() else 0
                size_str = f"{size/1024:.1f} KB"
                print(f"{path.name:<40} {size_str:<10} {'Found':<10}")
        
        if missing > 0:
            print(f"\nMissing files:")
            for mesh in resolved:
                if not mesh['exists']:
                    print(f"  - {mesh['file']} (from {mesh['link']})")


def parse_xml_and_extract_stls(xml_file: str) -> Tuple[List[str], Dict]:
    """
    Convenience function to parse XML and get STL files.
    
    Args:
        xml_file: Path to XML file
        
    Returns:
        (stl_files, metadata) tuple
    """
    parser = RobotMeshParser(xml_file)
    stl_files = parser.get_unique_stl_files()
    
    metadata = {
        'xml_file': str(parser.xml_file),
        'format': parser.format,
        'num_stl_files': len(stl_files),
        'mesh_info': parser.get_mesh_info()
    }
    
    return stl_files, metadata


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python xml_parser.py <path_to_xml>")
        print("\nExample:")
        print("  python xml_parser.py ../../../isaac_assets/SO_ARM100/so100.urdf")
        sys.exit(1)
    
    xml_file = sys.argv[1]
    
    try:
        parser = RobotMeshParser(xml_file)
        parser.print_summary()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




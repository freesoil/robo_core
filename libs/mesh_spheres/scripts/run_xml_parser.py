#!/usr/bin/env python3
"""
Wrapper script to run xml_parser as a module.
"""

import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
libs_dir = script_dir.parent.parent  # robo_core/libs
if str(libs_dir) not in sys.path:
    sys.path.insert(0, str(libs_dir))

from mesh_spheres.xml_parser import RobotMeshParser

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_xml_parser.py <xml_file>")
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


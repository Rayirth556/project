"""
project_tree_structure.py
=========================
Generate the current project tree structure.
"""

import os
from pathlib import Path

def generate_tree():
    """Generate project tree structure"""
    project_root = Path(".")
    
    print("PROJECT TREE STRUCTURE")
    print("=" * 80)
    
    def print_tree(path, prefix="", is_dir=True):
        """Recursively print tree structure"""
        try:
            items = sorted(path.iterdir())
            files = [item for item in items if item.is_file()]
            dirs = [item for item in items if item.is_dir()]
            
            # Print directories first
            for i, item in enumerate(dirs):
                is_last = (i == len(dirs) - 1) and len(files) == 0
                connector = "└── " if is_last else "├── "
                print(f"{prefix}{connector}{item.name}/")
                
                # Skip certain directories
                if item.name in ['.git', '__pycache__', '.pytest_cache']:
                    continue
                    
                # Recurse into directory
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, is_dir=True)
            
            # Then print files
            for i, item in enumerate(files):
                is_last = i == len(files) - 1
                connector = "└── " if is_last else "├── "
                
                # Only show relevant file types
                if item.suffix.lower() in ['.py', '.md', '.json', '.csv', '.txt', '.parquet']:
                    print(f"{prefix}{connector}{item.name}")
        
        except PermissionError:
            print(f"{prefix}└── [Permission Denied]")
    
    print_tree(project_root)

def generate_summary():
    """Generate project summary"""
    print("\n" + "=" * 80)
    print("PROJECT SUMMARY")
    print("=" * 80)
    
    # Count files by type
    project_root = Path(".")
    py_files = list(project_root.rglob("*.py"))
    md_files = list(project_root.rglob("*.md"))
    json_files = list(project_root.rglob("*.json"))
    
    print(f"Python Files: {len(py_files)}")
    print(f"Markdown Files: {len(md_files)}")
    print(f"JSON Files: {len(json_files)}")
    
    # Key directories
    dirs = [d for d in project_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    print(f"Key Directories: {len(dirs)}")
    
    for d in sorted(dirs):
        if d.name in ['pdr_pipeline', 'data', 'models', 'backend']:
            print(f"  DIR {d.name}/")

if __name__ == "__main__":
    generate_tree()
    generate_summary()

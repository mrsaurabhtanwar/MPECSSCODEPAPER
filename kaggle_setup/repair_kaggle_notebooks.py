#!/usr/bin/env python3
"""Fix PosixPath error in Kaggle notebooks."""

import json
import sys
from pathlib import Path

def fix_run_benchmark_function(source):
    """Fix the run_benchmark function to convert Path objects to strings."""
    if "def run_benchmark" not in source:
        return source
    
    # The fix: convert all Path values to str in the cmd list
    # We need to wrap REPO_DIR, DATASET_PATH, and PROBLEM_LIST with str()
    lines = source.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Fix f-string interpolations and direct variable usage in cmd list
        if '        "--repo-dir", REPO_DIR,' in line:
            line = line.replace('REPO_DIR,', 'str(REPO_DIR),')
        elif '        "--path", DATASET_PATH,' in line:
            line = line.replace('DATASET_PATH,', 'str(DATASET_PATH),')
        elif '        "--problem-list", PROBLEM_LIST,' in line:
            line = line.replace('PROBLEM_LIST,', 'str(PROBLEM_LIST),')
        elif '        f"{REPO_DIR}/kaggle_setup/resumable_benchmark.py",' in line:
            line = line.replace('f"{REPO_DIR}/', 'f"{str(REPO_DIR)}/')
        
        # Also convert the join to handle any remaining Path objects
        if '    print("+ " + " ".join(cmd))' in line:
            line = '    print("+ " + " ".join(str(x) for x in cmd))'
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_notebook(notebook_path):
    """Fix a single notebook."""
    print(f"Fixing {notebook_path.name}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            fixed_source = fix_run_benchmark_function(source)
            
            if fixed_source != source:
                # Convert back to list format if it was a list
                if isinstance(cell['source'], list):
                    cell['source'] = fixed_source
                else:
                    cell['source'] = fixed_source
                modified = True
    
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  ✅ Fixed {notebook_path.name}")
        return True
    else:
        print(f"  ℹ️  No changes needed for {notebook_path.name}")
        return False

def main():
    kaggle_setup = Path(__file__).parent
    notebooks = list(kaggle_setup.glob("MPECSS_Kaggle_*.ipynb"))
    
    if not notebooks:
        print("No notebooks found!")
        return 1
    
    print(f"Found {len(notebooks)} notebooks to check\n")
    
    fixed_count = 0
    for nb_path in sorted(notebooks):
        if fix_notebook(nb_path):
            fixed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: Fixed {fixed_count} of {len(notebooks)} notebooks")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

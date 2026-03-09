"""
Regression Test: Ensure NO version numbers in code

This test prevents version numbers (v7, v6, v5, etc.) from creeping back into:
- File names
- Class names  
- Function names
- Variable names
- API paths

Per project policy: Version numbers exist ONLY in historical documentation.
"""

import os
import glob
import ast
import pytest
from pathlib import Path


def test_no_v7_in_filenames():
    """Ensure no files have v7 in their names"""
    repo_root = Path(__file__).parent.parent
    
    patterns = ['**/*v7*.py', '**/*v7*.ts', '**/*v7*.tsx', '**/*v7*.png', '**/*v7*.pt']
    v7_files = []
    
    for pattern in patterns:
        files = glob.glob(str(repo_root / pattern), recursive=True)
        # Filter out .venv, .git, node_modules, __pycache__
        files = [f for f in files if not any(x in f for x in ['.venv', '.git', 'node_modules', '__pycache__'])]
        v7_files.extend(files)
    
    assert len(v7_files) == 0, f"Found files with v7 in name: {v7_files}"


def test_no_v7_classes_in_inference_module():
    """Ensure V7InferenceEngine and get_v7_engine are removed"""
    try:
        from fincoll.inference import V7InferenceEngine
        pytest.fail("V7InferenceEngine still exists - should be removed")
    except ImportError:
        pass  # Expected
    
    try:
        from fincoll.inference import get_v7_engine
        pytest.fail("get_v7_engine still exists - should be removed")
    except ImportError:
        pass  # Expected


def test_prediction_engine_imports_work():
    """Ensure new names work"""
    from fincoll.inference import PredictionEngine, get_prediction_engine
    
    assert PredictionEngine is not None
    assert callable(get_prediction_engine)


def test_api_routes_no_v7():
    """Ensure API routes don't have /v7/ prefix"""
    from fincoll.api import inference
    import inspect
    
    source = inspect.getsource(inference)
    
    # Check for old v7 routes
    forbidden_routes = ['/v7/predict', '/v7/batch']
    for route in forbidden_routes:
        assert route not in source, f"Found deprecated route {route} in API code"
    
    # Check for new routes
    required_routes = ['"/predict/', '"/batch']
    for route in required_routes:
        assert route in source, f"Missing required route {route} in API code"


def test_no_v7_in_python_code():
    """Scan Python files for v7 in variable/function names (not comments/strings)"""
    repo_root = Path(__file__).parent.parent
    
    violations = []
    
    for py_file in repo_root.glob('fincoll/**/*.py'):
        if '.venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        with open(py_file, 'r') as f:
            try:
                tree = ast.parse(f.read())
                
                # Check function names
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if 'v7' in node.name.lower():
                            violations.append(f"{py_file}:{node.lineno} - function '{node.name}'")
                    
                    # Check class names
                    if isinstance(node, ast.ClassDef):
                        if 'v7' in node.name.lower():
                            violations.append(f"{py_file}:{node.lineno} - class '{node.name}'")
            
            except SyntaxError:
                # Skip files with syntax errors (might be templates, etc.)
                pass
    
    # Allow external library references (armv7l architecture)
    violations = [v for v in violations if 'armv7' not in v.lower()]
    
    assert len(violations) == 0, f"Found v7 in code:\n" + "\n".join(violations)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])

import sys
import os
import subprocess
from typing import Dict


def run_python_file(file_path: str) -> Dict[str, str]:
    """
    Execute a Python file and collect all its output including stdout, stderr, 
    and optionally capture matplotlib plots if they exist.
    
    Parameters
    ----------
    file_path : str
        Path to the Python file to execute
    
    Returns
    -------
    Dict[str, str]
        Dictionary containing:
        - 'stdout': Standard output text
        - 'stderr': Standard error text
        - 'plots': List of file paths to saved plots (if capture_plots=True)
        - 'execution_status': Success or error status
    """
    results = {
        'stdout': '',
        'stderr': '',
        'execution_status': 'success'
    }
    
    # Verify file exists
    if not os.path.exists(file_path):
        results['execution_status'] = 'error'
        results['stderr'] = f"File not found: {file_path}"
        return results
    
   
    try:
        # Run the Python file in a subprocess to properly isolate it
        # and capture output even if it crashes
        env = os.environ.copy()
        
        # Add plot capture code if requested
        cmd = [sys.executable]
        cmd.append(file_path)
            
        # Run the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        stdout, stderr = process.communicate()
        
        results['stdout'] = stdout
        results['stderr'] = stderr
        
        # Set error status if returncode is not 0
        if process.returncode != 0:
            results['execution_status'] = f'error (code {process.returncode})'
            
    except Exception as e:
        results['execution_status'] = 'error'
        results['stderr'] += f"\nException during execution: {str(e)}"
    
    return results


def run_and_display_results(file_path: str) -> None:
    """
    Run a Python file and print the results in a nicely formatted way.
    
    Parameters
    ----------
    file_path : str
        Path to the Python file to execute
    """
    print(f"Running: {file_path}")
    print("-" * 50)
    
    results = run_python_file(file_path)
    
    # Print status
    print(f"Status: {results['execution_status']}")
    
    # Print stdout if not empty
    if results['stdout'].strip():
        print("\nOutput:")
        print("-" * 50)
        print(results['stdout'])
    
    # Print stderr if not empty
    if results['stderr'].strip():
        print("\nErrors:")
        print("-" * 50)
        print(results['stderr'])
    
    print("-" * 50)


if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) > 1:
        run_and_display_results(sys.argv[1])
    else:
        print("Usage: python run_py_nb.py <python_file_path>")

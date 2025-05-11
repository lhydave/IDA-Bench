import sys
import os
import subprocess
import json
from typing import Dict


def run_python_file(file_path: str, output_json_path: str) -> None:
    """
    Execute a Python file and collect all its output including stdout, stderr, 
    and optionally capture matplotlib plots if they exist. Save results to a JSON file.
    
    Parameters
    ----------
    file_path : str
        Path to the Python file to execute
    output_json_path : str
        Path where the JSON results file should be saved
    """
    results = {
        'stdout': '',
        'stderr': '',
        'execution_status': 'success',
        'value_from_execution': None  # Will store numerical value from execution
    }
    
    # Verify file exists
    if not os.path.exists(file_path):
        results['execution_status'] = 'error'
        results['stderr'] = f"File not found: {file_path}"
        # Save results even if file not found
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        return
    
   
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
        
        # Try to extract numerical value from stdout
        try:
            # Look for the last line that contains a number
            lines = stdout.strip().split('\n')
            for line in reversed(lines):
                # Try to convert the line to a float
                try:
                    value = float(line.strip())
                    results['value_from_execution'] = value
                    break
                except ValueError:
                    continue
        except Exception as e:
            # If we can't extract a value, leave it as None
            pass
        
        # Set error status if returncode is not 0
        if process.returncode != 0:
            results['execution_status'] = f'error (code {process.returncode})'
            
    except Exception as e:
        results['execution_status'] = 'error'
        results['stderr'] += f"\nException during execution: {str(e)}"
    
    # Save results to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)


def run_and_display_results(file_path: str, output_json_path: str) -> None:
    """
    Run a Python file and print the results in a nicely formatted way.
    
    Parameters
    ----------
    file_path : str
        Path to the Python file to execute
    output_json_path : str
        Path where the JSON results file should be saved
    """
    print(f"Running: {file_path}")
    print("-" * 50)
    
    run_python_file(file_path, output_json_path)
    
    # Load and display results from JSON file
    with open(output_json_path, 'r') as f:
        results = json.load(f)
    
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
    
    # Print value from execution if found
    if results['value_from_execution'] is not None:
        print("\nValue from execution:")
        print("-" * 50)
        print(results['value_from_execution'])
    
    print("-" * 50)


if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) > 2:
        run_and_display_results(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python run_py_nb.py <python_file_path> <output_json_path>")

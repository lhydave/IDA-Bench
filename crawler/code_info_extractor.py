import re
import ast
import os
import nbformat
from typing import Literal
from data_manager.kaggle_info import CodeInfo
from crawler.utils import check_filter_keywords
from logger import logger


def extract_code_from_notebook(notebook_path: str) -> str:
    """
    Extract code from a Jupyter notebook file.

    Args:
        notebook_path: Path to the ipynb file

    Returns:
        A string containing all code cells concatenated
    """
    logger.info(f"Starting to extract code from notebook: {notebook_path}")
    with open(notebook_path, encoding="utf-8") as file:
        notebook = nbformat.read(file, as_version=4)

    code_cells = [cell["source"] for cell in notebook.cells if cell["cell_type"] == "code"]
    code = "\n".join(code_cells)
    logger.info(f"Successfully extracted {len(code_cells)} code cells from {notebook_path}")
    return code


def remove_comments_and_strings(code: str) -> str:
    """
    Remove comments and string literals from the code using AST.

    Args:
        code: Python code as string

    Returns:
        Code with comments and string literals removed
    """
    logger.info("Starting to remove comments and strings from code")
    try:
        # Parse the code into an AST
        tree = ast.parse(code)

        # Create a code transformer that replaces string literals with empty strings
        class StringAndDocstringRemover(ast.NodeTransformer):
            def visit_Constant(self, node):
                # Handle Python 3.8+
                if isinstance(node.value, str):
                    return ast.Constant(value="")
                return node

            def visit_Expr(self, node):
                # Check if expression is a string (docstring)
                if (
                    hasattr(ast, "Constant")
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    # Replace docstrings with empty expressions
                    return None
                return self.generic_visit(node)

        # Apply the transformer
        transformed_tree = StringAndDocstringRemover().visit(tree)
        ast.fix_missing_locations(transformed_tree)

        # Generate code from the modified AST
        processed_code = ast.unparse(transformed_tree)
        logger.info("Successfully removed comments and strings from code")
        return processed_code
    except SyntaxError:
        # If code has syntax errors that prevent AST parsing, return the original code
        # This ensures we don't lose the ability to analyze syntactically incorrect code
        logger.warning("Syntax error in code, could not fully process comments and strings, returning original code")
        return code


def count_patterns(code: str) -> dict:
    """
    Count occurrences of various patterns in the code.

    Args:
        code: The code to analyze

    Returns:
        Dictionary with counts of different patterns
    """
    logger.info("Starting to count code patterns")
    patterns = {
        "num_pivot_table": r"\.pivot_table\(|pivot_table\(",
        "num_groupby": r"\.groupby\(",
        "num_apply": r"\.apply\(",
        "num_def": r"\bdef\s+",
        "num_for": r"\bfor\s+",
        "num_and": r"\b\&\b",
        "num_or": r"\b\|b",
        "num_merge": r"\.merge\(",
        "num_concat": r"\.concat\(|pd\.concat\(",
        "num_join": r"\.join\(",
        "num_agg": r"\.agg\(|\.aggregate\(",
    }

    counts = {}
    for key, pattern in patterns.items():
        counts[key] = len(re.findall(pattern, code))

    logger.info("Finished counting code patterns")
    return counts


def extract_imports(code: str) -> list[str]:
    """
    Extract import statements from the code using AST.

    Args:
        code: Python code as string

    Returns:
        List of imported modules
    """
    logger.info("Starting to extract imports from code")
    imports = []

    try:
        # Parse the code into an AST
        tree = ast.parse(code)

        # Walk through the AST to find import statements
        for node in ast.walk(tree):
            # Handle 'import x' and 'import x.y.z' statements
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Get the root module (e.g., 'numpy' from 'numpy.random')
                    root_module = name.name.split(".")[0]
                    if root_module not in imports:
                        imports.append(root_module)

            # Handle 'from x import y' statements
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                # Get the root module (e.g., 'pandas' from 'pandas.core')
                root_module = node.module.split(".")[0]
                if root_module not in imports:
                    imports.append(root_module)

        logger.info(f"Successfully extracted {len(imports)} imported modules")
        return imports
    except SyntaxError:
        logger.error("Syntax error in code, cannot resolve imports")
        raise


def extract_code_info(notebook_path: str) -> CodeInfo | Literal["Keyword found"]:
    """
    Extract code information from a Jupyter notebook.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        "Keyword found" if filtered keywords are detected, or CodeInfo object
    """
    logger.info(f"Starting to extract code info from notebook: {notebook_path}")
    # Extract code from notebook
    code = extract_code_from_notebook(notebook_path)

    # Check if the notebook should be filtered
    if check_filter_keywords(code):
        logger.info(f"Filtered keywords found in {notebook_path}, skipping analysis")
        return "Keyword found"

    # Get original code size
    file_size = os.path.getsize(notebook_path)
    pure_code_size = len(code.encode("utf-8"))
    logger.debug(f"File size: {file_size} bytes, pure code size: {pure_code_size} bytes")

    # Remove comments and strings for analysis
    clean_code = remove_comments_and_strings(code)

    # Count patterns
    counts = count_patterns(clean_code)

    # Extract imports
    imports = extract_imports(code)

    # Create CodeInfo
    code_info = CodeInfo(**counts, import_list=imports, file_size=file_size, pure_code_size=pure_code_size)
    logger.info(f"Successfully extracted code info from {notebook_path}")

    return code_info

import re
import ast
import os
import nbformat
from typing import Literal
from data_manager import CodeInfo
from crawler.utils import check_filter_keywords
from logger import logger


def extract_code_and_count_features(notebook_path: str) -> tuple[str, int, int]:
    """
    Extract code from a Jupyter notebook file and count code cells and count number of 'feature' literals.

    Args:
        notebook_path: Path to the ipynb file

    Returns:
        A tuple containing:
            - A string with all code cells concatenated
            - An integer representing the number of code cells
            - An integer representing the count of 'feature' word occurrences
    """
    logger.info(f"Starting to extract code from notebook: {notebook_path}")
    with open(notebook_path, encoding="utf-8") as file:
        notebook = nbformat.read(file, as_version=4)

    code_cells = [cell["source"] for cell in notebook.cells if cell["cell_type"] == "code"]
    num_code_cells = len(code_cells)
    code = "\n".join(code_cells)

    # Count occurrences of "feature" in all cells (both code and markdown)
    all_cell_content = "\n".join(cell["source"] for cell in notebook.cells)

    # Count feature word occurrences (case insensitive)
    num_feature = len(re.findall(r"\bfeature(:?s?)\b", all_cell_content, re.IGNORECASE))

    logger.info(
        f"Successfully extracted {num_code_cells} code cells and found {num_feature} occurrences of 'feature' in {notebook_path}"  # noqa: E501
    )
    return code, num_code_cells, num_feature


def remove_comments_and_strings(code: str) -> str:
    """
    Remove comments. Replace all string literals and docstrings with empty strings.

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
                if isinstance(node.value, str):
                    return ast.Constant(value="")
                return node

        # Apply the transformer
        transformed_tree = StringAndDocstringRemover().visit(tree)

        # Generate code from the modified AST
        processed_code = ast.unparse(transformed_tree)
        logger.info("Successfully removed comments and strings from code")
        return processed_code
    except SyntaxError:
        # If code has syntax errors that prevent AST parsing, return the original code
        # This ensures we don't lose the ability to analyze syntactically incorrect code
        logger.warning("Syntax error in code, could not fully process comments and strings, returning original code")
        return code


def count_patterns(code: str) -> dict[str, int]:
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
        "num_and": r"(?<!&)&(?!&)",
        "num_or": r"(?<!\|)\|(?!\|)",
        "num_merge": r"\.merge\(",
        "num_concat": r"\.concat\(|pd\.concat\(",
        "num_join": r"\.join\(",
        "num_agg": r"\.agg\(|\.aggregate\(",
    }

    counts: dict[str, int] = {}
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
    imports = set[str]()

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
                        imports.add(root_module)

            # Handle 'from x import y' statements
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                # Get the root module (e.g., 'pandas' from 'pandas.core')
                root_module = node.module.split(".")[0]
                if root_module not in imports:
                    imports.add(root_module)

        logger.info(f"Successfully extracted {len(imports)} imported modules")
        return list(imports)
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
    code, num_python_cells, num_feature = extract_code_and_count_features(notebook_path)

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
    code_info = CodeInfo(
        **counts,
        import_list=imports,
        file_size=file_size,
        pure_code_size=pure_code_size,
        num_feature=num_feature,
        num_python_cells=num_python_cells,
    )
    logger.info(f"Successfully extracted code info from {notebook_path}")

    return code_info

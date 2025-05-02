import re
import ast
import os
import nbformat
from typing import Literal
from data_manager import CodeInfo, NotebookManager
from nbconvert import PythonExporter
from crawler.utils import check_filter_keywords
from logger import logger


def extract_code_and_count_features(notebook_path: str) -> tuple[str, int, int]:
    """
    Extract code from a Jupyter notebook file and count code cells and count number of 'feature' literals.
    Uses nbconvert module to efficiently convert the notebook to Python script.

    Args:
        notebook_path: Path to the ipynb file

    Returns:
        A tuple containing:
            - A string with all code cells concatenated
            - An integer representing the number of code cells
            - An integer representing the count of 'feature' word occurrences
    """
    logger.info(f"Starting to extract code from notebook: {notebook_path}")

    # Read the original notebook to count cells and features
    with open(notebook_path, encoding="utf-8") as file:
        notebook = nbformat.read(file, as_version=4)
    # Count number of code cells
    num_code_cells = sum(1 for cell in notebook.cells if cell["cell_type"] == "code")

    # Count occurrences of "feature" in all cells (both code and markdown)
    all_cell_content = "\n".join(cell["source"] for cell in notebook.cells)
    num_feature = len(re.findall(r"\bfeature(:?s?)\b", all_cell_content, re.IGNORECASE))

    # Use nbconvert module directly instead of subprocess
    try:
        # Extract only code cells from the notebook first to avoid nbconvert bugs
        code_cells = [cell for cell in notebook.cells if cell["cell_type"] == "code"]

        # Create a new notebook with only code cells
        code_only_notebook = nbformat.v4.new_notebook()
        code_only_notebook.cells = code_cells

        # Use PythonExporter to convert the code-only notebook to Python code
        python_exporter = PythonExporter()
        code, _ = python_exporter.from_notebook_node(code_only_notebook)

        logger.info(
            f"Successfully extracted {num_code_cells} code cells and found {num_feature} occurrences of 'feature' in {notebook_path}"  # noqa: E501
        )
        return code, num_code_cells, num_feature

    except Exception as e:
        logger.error(f"Failed to convert notebook using nbconvert module: {e}")

        # Fallback to manual extraction method if nbconvert fails
        logger.info("Falling back to manual extraction method")
        code_cells = []
        for cell in notebook.cells:
            if cell["cell_type"] == "code":
                # Remove magic commands (lines starting with % or %%)
                lines = cell["source"].split("\n")
                filtered_lines = [line for line in lines if not line.strip().startswith(("%", "%%", "!"))]
                code_cells.append("\n".join(filtered_lines))

        code = "\n".join(code_cells)

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
        "num_plots": r"\.show\(",
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


def update_all_code_info(notebook_manager: NotebookManager, do_filter: bool = True):
    """
    Extract code information from all notebooks in the given notebook manager and update their metadata.

    Args:
        notebook_manager: The notebook manager containing notebooks to process
        do_filter: Whether to filter notebooks based on keywords
    """
    logger.info("Starting to extract code info from all notebooks")

    total_count = 0
    success_count = 0
    filtered_count = 0
    failure_count = 0

    # Get all notebook IDs from the manager
    notebook_ids = notebook_manager.kept_list_index.copy()

    for notebook_id in notebook_ids:
        total_count += 1

        try:
            # Get notebook info
            notebook_info = notebook_manager.get_meta_info(notebook_id)

            if not notebook_info:
                logger.warning(f"No metadata found for notebook {notebook_id}, skipping")
                failure_count += 1
                continue

            # Check if the notebook has a local path
            if not notebook_info.path or not os.path.exists(notebook_info.path):
                logger.warning(f"Notebook {notebook_id} has no valid local path, skipping")
                failure_count += 1
                continue

            # Extract code info
            code_info = extract_code_info(notebook_info.path)

            # If filtered out and filtering is enabled
            if do_filter and code_info == "Keyword found":
                logger.info(f"Notebook {notebook_id} filtered out due to keywords")
                notebook_manager.remove_notebook(notebook_id, code_info)  # type: ignore
                filtered_count += 1
                continue

            # If extraction was successful
            if isinstance(code_info, CodeInfo):
                # Update notebook info with code info
                notebook_info.code_info = code_info
                notebook_manager.update_meta_info(notebook_id, notebook_info)
                success_count += 1
                if total_count % 100 == 0:
                    logger.info(f"Processed {total_count} notebooks, {success_count} successful extractions")
            else:
                logger.warning(f"Failed to extract code info from notebook {notebook_id}")
                failure_count += 1

        except Exception as e:
            logger.error(f"Error processing notebook {notebook_id}: {str(e)}")
            failure_count += 1

    logger.info(
        f"Code info extraction completed: {success_count}/{total_count} successful, "
        f"{filtered_count} filtered, {failure_count} failed"
    )

import re
from pathlib import Path


def set_data_dir(pyfile: str | Path, new_prefix: str, *, outfile: str | Path | None = None) -> str:
    """Replace every hard‑coded ``*.csv`` path in *pyfile* so that its directory part is
    replaced by *new_prefix*.

    Parameters
    ----------
    pyfile : str | Path
        Path to the Python script whose text will be modified.
    new_prefix : str
        Directory prefix that should precede each ``.csv`` file name, e.g. ``"data/test"``.
        A trailing slash is optional; it will be normalised.
    outfile : str | Path | None, optional
        Where to write the modified script.  *None* (default) overwrites *pyfile* in place.

    Returns
    -------
    str
        The full modified source code (also written to disk).

    Examples
    --------
    >>> set_data_dir('titanic_analysis.py', 'data/test')
    '...modified source...'
    """
    # --- Normalise inputs --------------------------------------------------
    new_prefix = new_prefix.rstrip("/\\")  # remove any trailing separator
    pyfile = Path(pyfile)
    if outfile is None:
        outfile = pyfile
    else:
        outfile = Path(outfile)

    # --- Read the original file -------------------------------------------
    original_code = pyfile.read_text(encoding="utf‑8")

    # --- Regex to capture any string literal ending in .csv ----------------
    #   1. (?P<quote>['"])        captures the opening quote (single or double)
    #   2. (?P<path>.*?)          non‑greedy dir path (may include escaped chars)
    #   3. (?P<file>[^/\\]+\.csv) the file name ending with .csv
    #   4. (?P=quote)             matching closing quote
    csv_regex = re.compile(r"(?P<quote>['\"])(?P<path>(?:[^'\"\\]|\\.)*?/)?(?P<file>[^/'\"\\]+\.csv)(?P=quote)")

    # --- Replacement function ---------------------------------------------
    def _replace(match: re.Match) -> str:
        quote = match.group('quote')
        file_name = match.group('file')
        new_path = f"{new_prefix}/{file_name}"
        return f"{quote}{new_path}{quote}"

    modified_code = csv_regex.sub(_replace, original_code)

    # --- Write back --------------------------------------------------------
    outfile.write_text(modified_code, encoding="utf‑8")
    return modified_code

def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python set_data_dir.py <python_file> <new_prefix>")
        sys.exit(1)
    else:
        py_file_path = sys.argv[1]
        new_prefix = sys.argv[2]
        set_data_dir(py_file_path, new_prefix)

# Example usage:
if __name__ == "__main__":
    main()
def to_filename(ID: str, replace_slash: bool = True) -> str:
    """
    Convert an ID (like xxx/yyyy) into a filename.
    If replace_slash is True, replace all '/' with '#####'.

    Args:
        ID: The ID to convert
        replace_slash: Whether to replace '/' with '#####'

    Returns:
        The filename representation
    """
    if replace_slash:
        return ID.replace("/", "#####")
    return ID


def notebook_id_to_url(notebook_id: str) -> str:
    """
    Convert a notebook ID to its URL on Kaggle.
    Args:
        notebook_id: The notebook ID to convert.
    Returns:
        str: The URL of the notebook.
    """
    return f"https://www.kaggle.com/code/{notebook_id}"


def url_to_notebook_id(notebook_url: str) -> str:
    """
    Extract notebook ID from a Kaggle notebook URL.
    Args:
        notebook_url: The URL of the notebook.
    Returns:
        str: The notebook ID.
    """
    # Extract the part after "/code/" and before "/notebook" if it exists
    if "/notebook" in notebook_url:
        return "/".join(notebook_url.split("/code/")[1].split("/notebook")[0].split("/"))
    # Otherwise, just extract the part after "/code/"
    return "/".join(notebook_url.split("/code/")[1].split("/"))

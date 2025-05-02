import numpy as np


def topk(k: int, total_count: int) -> list[int]:
    """
    Select the top k indices from a sorted list.

    Args:
        k (int): Number of elements to sample.
        total_count (int): Total number of elements available.

    Returns:
        List[int]: List of selected indices.
    """
    if k >= total_count:
        return list(range(total_count))
    return list(range(k))


def uniform(k: int, total_count: int) -> list[int]:
    """
    Select k indices uniformly distributed across a list.

    Args:
        k (int): Number of elements to sample.
        total_count (int): Total number of elements available.

    Returns:
        List[int]: List of selected indices.
    """
    if k >= total_count:
        return list(range(total_count))

    # Calculate indices for uniform sampling
    indices = np.linspace(0, total_count - 1, num=k, dtype=int)
    return indices.tolist()  # type: ignore

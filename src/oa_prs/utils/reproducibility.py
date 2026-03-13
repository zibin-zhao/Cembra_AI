"""
Reproducibility utilities: seeding, hashing, environment logging.
"""

import hashlib
import platform
import random
from pathlib import Path
from typing import Optional

import numpy as np


def set_all_seeds(seed: int) -> None:
    """
    Set random seeds for all major libraries for reproducibility.

    Sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (if available)
    - TensorFlow (if available)

    Parameters
    ----------
    seed : int
        Random seed value

    Examples
    --------
    >>> set_all_seeds(42)
    """
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # TensorFlow
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def compute_file_hash(
    path: str | Path,
    algo: str = "sha256",
) -> str:
    """
    Compute cryptographic hash of file contents.

    Parameters
    ----------
    path : str | Path
        Path to file
    algo : str
        Hash algorithm (sha256, md5, sha1, etc.)

    Returns
    -------
    str
        Hex digest of file hash

    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If algorithm is not supported

    Examples
    --------
    >>> hash_val = compute_file_hash("data.parquet")
    >>> print(f"SHA256: {hash_val}")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        hash_obj = hashlib.new(algo)
    except ValueError as e:
        raise ValueError(f"Unknown hash algorithm: {algo}") from e

    # Read file in chunks for memory efficiency
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def log_environment() -> dict:
    """
    Capture environment and dependency versions.

    Returns
    -------
    dict
        Dictionary with environment information including:
        - python_version
        - platform (OS)
        - numpy_version
        - pandas_version
        - torch_version (if available)
        - tensorflow_version (if available)

    Examples
    --------
    >>> env = log_environment()
    >>> print(f"Python: {env['python_version']}")
    """
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    # Core dependencies
    try:
        import numpy

        env_info["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    try:
        import pandas

        env_info["pandas_version"] = pandas.__version__
    except ImportError:
        pass

    # Optional ML frameworks
    try:
        import torch

        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    try:
        import tensorflow

        env_info["tensorflow_version"] = tensorflow.__version__
    except ImportError:
        pass

    try:
        import sklearn

        env_info["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass

    return env_info

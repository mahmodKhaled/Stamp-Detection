import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List

def get_path_to(
    *args,
    repo_name: str = 'Stamp-Detection',
) -> str:
    """
    This function returns the path to any file or directory.

    Parameters
    ----------
    repo_name : str
        Name of the repository.
    *args : str
        Variable number of path components (folders/files) to join.
    
    Returns
    -------
    path : str
        Full path to the requested location.
    """
    base_dir = os.path.dirname(os.getcwd())
    if repo_name not in base_dir:
        base_dir = os.path.join(base_dir, repo_name)
        
    path = os.path.join(base_dir, *args)
    
    # Create parent directory if it doesn't exist
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    return path

def show_image(
    img: np.ndarray,
    showAxis: bool = False,
    size: Tuple[int, int] = (20, 10)
) -> None:
    plt.figure(figsize=size)
    if not showAxis:
        plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img[:,:,::-1])
    else:
        plt.imshow(img, cmap='gray')

def get_index_of_image_from_name(
    name: str,
    image_names: List[str]
) -> int:
    return image_names.index(name)

def imread(
    path: str
) -> np.ndarray:
    img = plt.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

import os

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

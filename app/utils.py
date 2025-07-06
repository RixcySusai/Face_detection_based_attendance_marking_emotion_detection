import os

def ensure_folder_exists(folder):
    """Ensures a folder exists."""
    if not os.path.exists(folder):
        os.makedirs(folder)

"""
Download Amazon ESCI Shopping Queries dataset: clone repo and pull LFS. Data lives in the clone.
"""
from __future__ import annotations  # Enable postponed evaluation of type hints

import subprocess  # For running shell commands (git clone, git lfs)
import sys  # For exit codes and stderr
from pathlib import Path  # For path manipulation

REPO_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels from src/data/download_esci.py to project root
DATA_DIR = REPO_ROOT / "data"  # Path to data directory
CLONE_DIR = DATA_DIR / "esci-data"  # Directory where git repo will be cloned
ESCI_DATA_DIR = CLONE_DIR / "shopping_queries_dataset"  # Path to parquet files inside clone
GIT_REPO = "https://github.com/amazon-science/esci-data.git"  # GitHub repository URL
BRANCH = "main"  # Git branch to clone


def main() -> int:
    """
    Clone ESCI repo and pull LFS files. Return 0 on success, 1 on error.
    """
    # Check if data already exists (both directory and examples parquet file)
    if ESCI_DATA_DIR.exists() and (ESCI_DATA_DIR / "shopping_queries_dataset_examples.parquet").exists():
        print(f"Already present at {ESCI_DATA_DIR}")  # Skip download if already present
        return 0  # Exit successfully
    DATA_DIR.mkdir(parents=True, exist_ok=True)  # Create data directory if it doesn't exist
    # Clone repository: shallow clone (--depth 1) of main branch to CLONE_DIR
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, GIT_REPO, str(CLONE_DIR)],  # Git clone command
        check=True,  # Raise exception if command fails
        cwd=REPO_ROOT,  # Run from project root
    )
    # Pull Git LFS files (large parquet files are stored in Git LFS)
    subprocess.run(["git", "lfs", "pull"], cwd=CLONE_DIR, capture_output=True)  # Download LFS files, suppress output
    # Verify that examples parquet file exists after clone and LFS pull
    if not (ESCI_DATA_DIR / "shopping_queries_dataset_examples.parquet").exists():
        print("Examples parquet not found. You may need to run 'git lfs pull' in the clone or download from the dataset page.", file=sys.stderr)  # Print error message
        return 1  # Exit with error code
    print(f"ESCI data ready at {ESCI_DATA_DIR}")  # Print success message
    return 0  # Exit successfully


if __name__ == "__main__":
    sys.exit(main())  # Run main and exit with return code

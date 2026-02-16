from __future__ import annotations
from pathlib import Path
from typing import Any
from kedro.io import AbstractDataset

class FolderDataset(AbstractDataset[Path, Path]):
    """
    A Kedro dataset that returns the Path to a file or directory.
    Perfect for PartitionedDatasets where the node needs the file path 
    to perform custom operations (like unzipping).
    """
    def __init__(self, filepath: str, **kwargs: Any):
        self._filepath = Path(filepath)
        # Store metadata (like suffix) for the _describe method
        self._metadata = kwargs

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _load(self) -> Path:
        # Simply return the path object to the node
        return self._filepath

    def _save(self, data: Any = None) -> None:
        # Ensure the directory exists; ignore 'data' if passed
        self._filepath.mkdir(parents=True, exist_ok=True)

    def _describe(self) -> dict[str, Any]:
        return dict(filepath=str(self._filepath), **self._metadata)

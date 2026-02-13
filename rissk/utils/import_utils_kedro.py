from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import re

from loguru import logger

from rissk.utils.import_utils import extract_zip


def _extract_path_candidates(partition_id: str, partition_loader: Any = None) -> List[Path]:
    partition_path = Path(partition_id)
    candidates = [partition_path, Path.cwd() / partition_path]

    loader = partition_loader
    if callable(loader):
        bound_loader = getattr(loader, "__self__", None)
        if bound_loader is not None:
            loader = bound_loader

    potential_sources = [loader]

    closure = getattr(partition_loader, "__closure__", None)
    if closure:
        for cell in closure:
            potential_sources.append(cell.cell_contents)

    for source in potential_sources:
        if source is None:
            continue

        for attr in ("filepath", "_filepath", "path", "_path"):
            value = getattr(source, attr, None)
            if value:
                source_path = Path(value)
                candidates.extend([source_path, source_path.parent])

    unique_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        unique_candidates.append(candidate)

    return unique_candidates


def _resolve_existing_path(partition_id: str, partition_loader: Any = None) -> Optional[Path]:
    for candidate in _extract_path_candidates(partition_id, partition_loader):
        if candidate.exists():
            return candidate
    return None


def extract_all_zip_files(partitions: Dict[str, Callable[[], Any]], zip_password: str = None) -> None:
    """
    Extract all zip files referenced by Kedro partition IDs.

    - Keeps naming convention: folder name = zip filename without ``.zip``.
    - Delegates nested-zip handling to ``extract_zip``.
    - Procedural utility; returns no value.
    """
    if not partitions:
        logger.warning("No partitions found for zip extraction")
        return

    zip_files: List[Path] = []

    for partition_id, loader in partitions.items():
        if not str(partition_id).lower().endswith(".zip"):
            continue

        existing_path = _resolve_existing_path(partition_id, loader)
        if existing_path and existing_path.is_file() and existing_path.suffix.lower() == ".zip":
            zip_files.append(existing_path)
        else:
            logger.warning(f"Partition zip path not found on disk: {partition_id}")

    logger.info(f"Found {len(zip_files)} zip files from partition entries")

    for zip_file in zip_files:
        destination = zip_file.with_suffix("")
        extract_zip(zip_file, destination, password=zip_password)


def filter_matching_folders(partitions: Dict[str, Callable[[], Any]], questionnaires: List[dict]) -> List[Path]:
    # This is a folder only version of the filter_matching_zip_files function, which is used 
    # to find matching folders after extraction. The logic is the same, but it targets folders 
    # instead of zip files.
    """
    Return extracted folder paths matching questionnaire/version patterns.

    Matching logic mirrors legacy ``get_zip_files`` naming rules, but starts from
    partition IDs instead of a raw path string.
    """
    if not partitions:
        logger.warning("No partitions found while filtering extracted folders")
        return []

    matching_folders: List[Path] = []
    seen = set()

    for questionnaire in questionnaires:
        name = questionnaire.get("name")
        versions = questionnaire.get("VERSION", [])

        version_pattern = "|".join(map(str, versions))
        pattern = re.compile(rf"{name}_({version_pattern})_.*")

        for partition_id, loader in partitions.items():
            partition_path = Path(partition_id)
            candidate_name = partition_path.stem if partition_path.suffix.lower() == ".zip" else partition_path.name

            if not pattern.match(candidate_name):
                continue

            existing_partition_path = _resolve_existing_path(partition_id, loader)

            folder_candidates: List[Path] = []
            if existing_partition_path and existing_partition_path.suffix.lower() == ".zip":
                folder_candidates.append(existing_partition_path.with_suffix(""))
            elif existing_partition_path:
                folder_candidates.append(existing_partition_path)

            if not existing_partition_path and partition_path.suffix.lower() == ".zip":
                for candidate in _extract_path_candidates(partition_id, loader):
                    folder_candidates.append(candidate.with_suffix(""))

            for folder_path in folder_candidates:
                if folder_path.is_dir():
                    folder_str = str(folder_path)
                    if folder_str not in seen:
                        seen.add(folder_str)
                        matching_folders.append(folder_path)

    logger.info(f"Filtered {len(matching_folders)} matching folders from partition entries")
    return matching_folders

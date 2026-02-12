from pathlib import Path
from typing import List
import re

from loguru import logger

from rissk.utils.import_utils import extract_zip


def extract_all_zip_files(raw_path: Path, zip_password: str = None) -> None:
    """
    Extract all zip files found at the top level of ``raw_path``.

    - Keeps naming convention: folder name = zip filename without ``.zip``.
    - Delegates nested-zip handling to ``extract_zip``.
    - Procedural utility; returns no value.
    """
    if not raw_path.exists():
        logger.warning(f"Raw path does not exist: {raw_path}")
        return

    zip_files = [
        file_path
        for file_path in raw_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == ".zip"
    ]

    logger.info(f"Found {len(zip_files)} zip files in {raw_path}")

    for zip_file in zip_files:
        destination = zip_file.with_suffix("")
        extract_zip(zip_file, destination, password=zip_password)


def filter_matching_folders(raw_path: Path, questionnaires: List[dict]) -> List[Path]:
    # This is a folder only version of the filter_matching_zip_files function, which is used 
    # to find matching folders after extraction. The logic is the same, but it targets folders 
    # instead of zip files.
    """
    Return folder paths in ``raw_path`` that match questionnaire/version patterns.

    Matching logic mirrors legacy ``get_zip_files`` naming rules, but targets
    extracted folders to support scenarios where zip files are unavailable.
    """
    if not raw_path.exists():
        logger.warning(f"Raw path does not exist: {raw_path}")
        return []

    matching_folders: List[Path] = []

    for questionnaire in questionnaires:
        name = questionnaire.get("name")
        versions = questionnaire.get("VERSION", [])

        version_pattern = "|".join(map(str, versions))
        pattern = re.compile(rf"{name}_({version_pattern})_.*")

        matching_folders.extend(
            path
            for path in raw_path.iterdir()
            if path.is_dir() and pattern.match(path.name)
        )

    logger.info(f"Filtered {len(matching_folders)} matching folders from {raw_path}")
    return matching_folders
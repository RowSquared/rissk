"""Nodes for ingesting Survey Solutions export data."""
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from loguru import logger

# Import your existing utilities
from rissk.utils.import_utils import (
    extract_zip,
    get_survey_info,
    get_dataframes
)


def unzip_raw_surveys(
    raw_zip_files: Dict[str, callable],
    parameters: Dict
) -> None:
    """
    Extract zipped Survey Solutions exports.
    
    Handles:
    - Recursive unzipping (nested ZIPs)
    - Password-protected ZIPs (from credentials)
    - Mixed formats (.dta, .tab)
    
    Args:
        raw_zip_files: Dictionary of ZIP files from catalog (PartitionedDataset)
        parameters: Survey configuration from parameters.yml
        
    Side Effect:
        Extracts files to same directory as ZIP (removes .zip extension)
    """
    survey_name = parameters["name"]
    questionnaires = parameters["questionnaires"]
    
    # Filter ZIP files based on survey configuration
    questionnaire_names = [q["name"] for q in questionnaires]
    
    matching_files = [
        filename for filename in raw_zip_files.keys()
        if any(qname in filename for qname in questionnaire_names)
    ]
    
    logger.info(f"Found {len(matching_files)} ZIP files to extract: {matching_files}")
    
    for filename in matching_files:
        # Get the full path to the ZIP file
        zip_path = Path("data/10_RAW") / filename
        dest_path = zip_path.with_suffix('')  # Remove .zip extension
        
        logger.info(f"Extracting {filename} to {dest_path}")
        extract_zip(zip_path, dest_path)
    
    logger.success(f"Extraction complete. Extracted {len(matching_files)} surveys.")


def load_survey_dataframes(
    parameters: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load paradata, questionnaire, and microdata from extracted files.
    
    Handles:
    - Mixed file formats (.dta for Stata, .tab for tabular)
    - Variable name parsing from Survey Solutions structure
    - Multi-option/GPS/List question transformations
    
    Args:
        parameters: Survey configuration
        
    Returns:
        tuple: (paradata_df, questionnaire_df, microdata_df)
    """
    # Use the data path from catalog structure
    raw_data_dir = Path("data/10_RAW")
    
    # Scan extracted directories for survey info
    survey_paths = []
    if raw_data_dir.exists():
        for item in raw_data_dir.iterdir():
            if item.is_dir():
                survey_paths.append(item)
    
    if not survey_paths:
        raise FileNotFoundError(
            f"No extracted survey data found in {raw_data_dir}. "
            "Make sure to run the unzip_surveys_node first."
        )
    
    survey_info = get_survey_info(survey_paths)
    
    logger.info(f"Loading dataframes for surveys: {list(survey_info.keys())}")
    
    # Use your existing get_dataframes logic
    paradata_df, questionnaire_df, microdata_df = get_dataframes(survey_info)
    
    logger.info(f"Loaded - Paradata: {paradata_df.shape}, "
                f"Questionnaire: {questionnaire_df.shape}, "
                f"Microdata: {microdata_df.shape}")
    
    return paradata_df, questionnaire_df, microdata_df

from typing import Dict, List, Tuple
from pathlib import Path
import os
import pandas as pd
from loguru import logger
from rissk.utils.import_utils import get_zip_files, extract_zip, get_survey_info, get_dataframes

def unzip_survey_data_node(
    survey_name: str,
    raw_path_str: str,
    questionnaires: List[Dict],
    zip_password: str
) -> List[Path]:
    """
    Finds and extracts zips. Returns list of extracted project paths.
    Wraps import_utils.extract_zip.
    """
    raw_path = Path(raw_path_str)
    
    logger.info(f"Looking for zips in {raw_path} for {survey_name}")
    zip_files = get_zip_files(raw_path, survey_name, questionnaires)
    
    extracted_paths = []
    for zip_file in zip_files:
        project_path = zip_file.with_suffix('')
        extracted_paths.append(project_path)
        # Extract using the password argument
        extract_zip(zip_file, project_path, password=zip_password)
        
    return extracted_paths

def load_survey_data_node(survey_paths: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads dataframes from extracted folders.
    Wraps import_utils.get_dataframes which handles .dta/.tab logic.
    """
    logger.info(f"Processing survey info for {len(survey_paths)} paths")
    survey_info = get_survey_info(survey_paths)
    
    # Returns: paradata, questionnaire, microdata
    dfs_para, dfs_qnr, dfs_micro = get_dataframes(survey_info)
    
    return dfs_para, dfs_qnr, dfs_micro

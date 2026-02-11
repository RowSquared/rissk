from typing import Dict, List, Tuple
from pathlib import Path
import os
import pandas as pd
from loguru import logger
from rissk.utils.import_utils import (
    get_zip_files, 
    extract_zip, 
    get_survey_info, 
    get_questionnaire,
    get_paradata,
    get_microdata
)


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

## If I want to add the fallback logic for existing folders, I can modify the above function like this:
# ...existing code...
def unzip_survey_data_node(
    survey_name: str,
    raw_path_str: str,
    questionnaires: List[Dict],
    zip_password: str
) -> List[Path]:
    """
    Finds and extracts zips. Returns list of extracted project paths.
    If zips are missing but folders exist, returns those folders.
    Wraps import_utils.extract_zip.
    """
    raw_path = Path(raw_path_str)
    
    logger.info(f"Looking for data in {raw_path} for {survey_name}")
    
    # 1. Try to find zips
    zip_files = get_zip_files(raw_path, survey_name, questionnaires)
    
    extracted_paths = []
    
    if zip_files:
        logger.info(f"Found {len(zip_files)} zip files to extract.")
        for zip_file in zip_files:
            project_path = zip_file.with_suffix('')
            extracted_paths.append(project_path)
            # Extract using the password argument
            extract_zip(zip_file, project_path, password=zip_password)
    else:
        # 2. If no zips, look for existing directories matching the naming convention
        logger.info("No zip files found. Looking for existing unzipped folders.")
        import re
        
        for questionnaire in questionnaires:
            name = questionnaire.get('name')
            versions = questionnaire.get('VERSION', [])
            version_pattern = "|".join(map(str, versions))
            # Matches folder names like: questionnaire_version_...
            # Note: The regex mimics get_zip_files but without .zip extension
            pattern = re.compile(rf"{name}_({version_pattern})_.*")
            
            matching_dirs = [
                d for d in raw_path.iterdir() 
                if d.is_dir() and pattern.match(d.name)
            ]
            extracted_paths.extend(matching_dirs)
            
        if extracted_paths:
            logger.info(f"Found {len(extracted_paths)} existing unzipped folders.")
        else:
            logger.warning(f"No zip files or matching folders found in {raw_path}")

    return extracted_paths
# ...existing code...



def load_paradata_node(survey_paths: List[Path]) -> pd.DataFrame:
    """
    Loads paradata from extracted folders.
    Independent node that generates its own questionnaire reference.
    """
    logger.info(f"Processing paradata for {len(survey_paths)} paths")
    survey_info = get_survey_info(survey_paths)
    
    dfs_paradata = []
    
    for survey_questionnaire, questionnaires_details in survey_info.items():
        for questionnaires_version, file_paths in questionnaires_details.items():
            tabular_path = file_paths['Tabular']
            paradata_path = file_paths['Paradata']

            try:
                # We need the questionnaire map even for paradata processing
                df_questionnaires = get_questionnaire(tabular_path)
                df_paradata = get_paradata(paradata_path, df_questionnaires)
                
                dfs_paradata.append(df_paradata)
                logger.info(f"Loaded paradata for {survey_questionnaire} v{questionnaires_version}")
            except Exception as e:
                logger.error(f"Failed to load paradata for {survey_questionnaire} v{questionnaires_version}. Skipping. Error: {str(e)}")
                continue

    if not dfs_paradata:
        return pd.DataFrame()

    combined_df = pd.concat(dfs_paradata)
    combined_df.reset_index(drop=True, inplace=True)
    
    if 'answer_sequence' in combined_df.columns:
        combined_df['answer_sequence'] = combined_df['answer_sequence'].apply(str)
        
    return combined_df


def load_questionnaire_node(survey_paths: List[Path]) -> pd.DataFrame:
    """
    Loads questionnaire metadata from extracted folders.
    """
    logger.info(f"Processing questionnaires for {len(survey_paths)} paths")
    survey_info = get_survey_info(survey_paths)
    
    dfs_questionnaires = []
    
    for survey_questionnaire, questionnaires_details in survey_info.items():
        for questionnaires_version, file_paths in questionnaires_details.items():
            tabular_path = file_paths['Tabular']

            try:
                df_questionnaires = get_questionnaire(tabular_path)
                dfs_questionnaires.append(df_questionnaires)
                logger.info(f"Loaded questionnaire for {survey_questionnaire} v{questionnaires_version}")
            except Exception as e:
                logger.error(f"Failed to load questionnaire for {survey_questionnaire} v{questionnaires_version}. Skipping. Error: {str(e)}")
                continue

    if not dfs_questionnaires:
        return pd.DataFrame()

    combined_df = pd.concat(dfs_questionnaires)
    combined_df.reset_index(drop=True, inplace=True)
    
    if 'answer_sequence' in combined_df.columns:
        combined_df['answer_sequence'] = combined_df['answer_sequence'].apply(str)
        
    return combined_df


def load_microdata_node(survey_paths: List[Path]) -> pd.DataFrame:
    """
    Loads microdata (answers) from extracted folders.
    Independent node that generates its own questionnaire reference.
    """
    logger.info(f"Processing microdata for {len(survey_paths)} paths")
    survey_info = get_survey_info(survey_paths)
    
    dfs_microdata = []
    
    for survey_questionnaire, questionnaires_details in survey_info.items():
        for questionnaires_version, file_paths in questionnaires_details.items():
            tabular_path = file_paths['Tabular']

            try:
                # We need the questionnaire map for variable types and structure
                df_questionnaires = get_questionnaire(tabular_path)
                df_microdata = get_microdata(tabular_path, df_questionnaires)
                
                dfs_microdata.append(df_microdata)
                logger.info(f"Loaded microdata for {survey_questionnaire} v{questionnaires_version}")
            except Exception as e:
                logger.error(f"Failed to load microdata for {survey_questionnaire} v{questionnaires_version}. Skipping. Error: {str(e)}")
                continue

    if not dfs_microdata:
        return pd.DataFrame()

    combined_df = pd.concat(dfs_microdata)
    combined_df.reset_index(drop=True, inplace=True)
    
    if 'answer_sequence' in combined_df.columns:
        combined_df['answer_sequence'] = combined_df['answer_sequence'].apply(str)
        
    return combined_df

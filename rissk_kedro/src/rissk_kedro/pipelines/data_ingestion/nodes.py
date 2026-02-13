from pathlib import Path
from typing import Any, Callable, Dict, List
import pandas as pd
from loguru import logger
from rissk.utils.import_utils_kedro import extract_all_zip_files, filter_matching_folders
from rissk.utils.import_utils import get_survey_info, get_questionnaire, get_paradata, get_microdata


def extract_zip_files_node(survey_partitions: Dict[str, Callable[[], Any]], zip_password: str) -> None:
    """
    Extract zip files referenced by the survey partition dataset.
    Procedural node: extraction side-effect only.
    """
    logger.info(f"Extracting zip files from {len(survey_partitions)} partition entries")
    extract_all_zip_files(survey_partitions, zip_password=zip_password)


def filter_extracted_survey_paths_node(survey_partitions: Dict[str, Callable[[], Any]], questionnaires: List[Dict]) -> List[Path]:
    """
    Return extracted folder paths matching questionnaire/version patterns
    using survey partition entries.
    This node does not perform extraction.
    """
    logger.info(f"Collecting matching survey folders from {len(survey_partitions)} partition entries")
    return filter_matching_folders(survey_partitions, questionnaires)


def load_paradata_node(file_paths: List[Path]) -> pd.DataFrame:
    """
    Loads paradata from extracted folders.
    Independent node that generates its own questionnaire reference.
    """
    logger.info(f"Processing paradata for {len(file_paths)} paths")
    survey_info = get_survey_info(file_paths)
    
    dfs_paradata = []
    
    for survey_questionnaire, questionnaires_details in survey_info.items():
        for questionnaires_version, file_paths in questionnaires_details.items():
            tabular_path = file_paths.get('Tabular')
            paradata_path = file_paths.get('Paradata')

            if not tabular_path or not paradata_path:
                logger.warning(
                    f"Skipping paradata load for {survey_questionnaire} v{questionnaires_version}: "
                    f"missing required exports (Tabular={bool(tabular_path)}, Paradata={bool(paradata_path)})"
                )
                continue

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


def load_questionnaire_node(file_paths: List[Path]) -> pd.DataFrame:
    """
    Loads questionnaire metadata from extracted folders.
    """
    logger.info(f"Processing questionnaires for {len(file_paths)} paths")
    survey_info = get_survey_info(file_paths)
    
    dfs_questionnaires = []
    
    for survey_questionnaire, questionnaires_details in survey_info.items():
        for questionnaires_version, file_paths in questionnaires_details.items():
            tabular_path = file_paths.get('Tabular')

            if not tabular_path:
                logger.warning(
                    f"Skipping questionnaire load for {survey_questionnaire} v{questionnaires_version}: "
                    "missing Tabular export"
                )
                continue

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


def load_microdata_node(file_paths: List[Path]) -> pd.DataFrame:
    """
    Loads microdata (answers) from extracted folders.
    Independent node that generates its own questionnaire reference.
    """
    logger.info(f"Processing microdata for {len(file_paths)} paths")
    survey_info = get_survey_info(file_paths)
    
    dfs_microdata = []
    
    for survey_questionnaire, questionnaires_details in survey_info.items():
        for questionnaires_version, file_paths in questionnaires_details.items():
            tabular_path = file_paths.get('Tabular')

            if not tabular_path:
                logger.warning(
                    f"Skipping microdata load for {survey_questionnaire} v{questionnaires_version}: "
                    "missing Tabular export"
                )
                continue

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

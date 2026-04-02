from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from rissk.utils.import_utils_kedro import (
    extract_zip,
    filter_matching_folders,
    get_survey_info,
    get_questionnaire,
    get_paradata_raw,
    get_microdata_raw,
    merge_microdata_questionnaire
)
from rissk.feature_processing_kedro import make_index_col


def extract_zip_files_node(survey_zip_partitions: Dict[str, Callable[[], Path]], zip_password: str) -> None:
    """
    Node that iterates through partitions and triggers extraction.
    Note: The type hint shows the loader returns a Path.
    """
    if not survey_zip_partitions:
        logger.warning("No zip partitions found to extract.")
        return

    for partition_id, loader in survey_zip_partitions.items():
        # 1. LOAD THE PATH (This calls PathDataset._load)
        zip_path = loader()
        
        # 2. VALIDATE & EXTRACT
        if zip_path.suffix.lower() == ".zip" and zip_path.exists():
            destination = zip_path.with_suffix("")
            logger.info(f"Extracting partition [{partition_id}] from {zip_path}")
            extract_zip(zip_path, destination, password=zip_password)
        # else:
        #     logger.debug(f"Skipping non-zip partition: {partition_id}")


def filter_extracted_survey_paths_node(survey_partitions: Dict[str, Callable[[], Any]], questionnaires: List[Dict]) -> List[Path]:
    """
    Return extracted folder paths matching questionnaire/version patterns
    using survey partition entries.
    This node does not perform extraction.
    """
    lines = ["=" * 55, "  DATA INGESTION — Questionnaires to process", "=" * 55]
    for q in questionnaires:
        versions = ", ".join(str(v) for v in q.get("VERSION", []))
        lines.append(f"  • {q['name']}  |  versions: [{versions}]")
    lines.append("=" * 55)
    logger.info("\n" + "\n".join(lines))

    logger.info(f"Collecting matching survey folders from {len(survey_partitions)} partition entries")
    return filter_matching_folders(survey_partitions, questionnaires)


def load_paradata_node(file_paths: List[Path]) -> pd.DataFrame:
    """
    Loads raw paradata from extracted folders.
    No questionnaire metadata is merged at this stage; column splitting,
    timestamp parsing and version tagging are performed by get_paradata_raw.
    """
    logger.info(f"Processing raw paradata for {len(file_paths)} paths")
    survey_info = get_survey_info(file_paths)

    dfs_paradata = []

    for survey_questionnaire, questionnaires_details in survey_info.items():
        for questionnaires_version, file_paths_detail in questionnaires_details.items():
            paradata_path = file_paths_detail.get('Paradata')

            if not paradata_path:
                logger.warning(
                    f"Skipping paradata load for {survey_questionnaire} v{questionnaires_version}: "
                    "missing Paradata export"
                )
                continue

            try:
                df_paradata = get_paradata_raw(paradata_path)
                dfs_paradata.append(df_paradata)
                logger.info(f"Loaded raw paradata for {survey_questionnaire} v{questionnaires_version}")
            except Exception as e:
                logger.error(f"Failed to load paradata for {survey_questionnaire} v{questionnaires_version}. Skipping. Error: {str(e)}")
                continue

    if not dfs_paradata:
        return pd.DataFrame()

    combined_df = pd.concat(dfs_paradata)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df

def process_paradata_node(
    paradata_raw: pd.DataFrame,
    questionnaire: pd.DataFrame,
    parameters: Dict,
) -> pd.DataFrame:
    """
    Merges questionnaire metadata onto raw paradata, processes timestamps and
    interviewing flags, makes the index column, and filters to active interviewer
    events - producing the paradata_processed dataset consumed by feature creation.
    """
    paradata = paradata_raw.copy()

    # 1. Merge questionnaire metadata
    if not questionnaire.empty:
        q_columns = [
            'qnr_seq', 'variable_name', 'qtype', 'question_type',
            'answers', 'question_scope',
            'yes_no_view', 'is_filtered_combobox',
            'is_integer', 'cascade_from_question_id',
            'answer_sequence', 'n_answers', 'question_sequence',
            'qnr', 'qnr_version',
        ]
        q_columns = [c for c in q_columns if c in questionnaire.columns]
        paradata = paradata.merge(
            questionnaire[q_columns],
            how='left',
            left_on=['param', 'qnr', 'qnr_version'],
            right_on=['variable_name', 'qnr', 'qnr_version'],
        )

    # 2. Stringify answer_sequence for parquet serialization (matches legacy behaviour)
    if 'answer_sequence' in paradata.columns:
        paradata['answer_sequence'] = paradata['answer_sequence'].apply(str)

    # 3. Calculate f__answer_hour_set
    paradata['f__answer_hour_set'] = (
        paradata['timestamp_local'].dt.hour +
        paradata['timestamp_local'].dt.round('30min').dt.minute / 60
    )

    # 4. Calculate interviewing flag and filter to first-pass interviewer events
    events_split = ['RejectedBySupervisor', 'OpenedBySupervisor', 'OpenedByHQ', 'RejectedByHQ']
    paradata['flag'] = paradata['event'].isin(events_split)
    paradata['cumulative_flag'] = paradata.groupby('interview__id')['flag'].cumsum()
    paradata['interviewing'] = np.where(paradata['cumulative_flag'] > 0, False, True)
    paradata.drop(['flag', 'cumulative_flag'], axis=1, inplace=True)
    paradata = paradata[(paradata['interviewing'] == True) & (paradata['role'] == 1)].copy()

    # 5. Make index column
    paradata = make_index_col(paradata)

    # 6. Sort
    paradata.sort_values(['interview__id', 'order'], inplace=True)
    paradata.reset_index(drop=True, inplace=True)

    return paradata


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
    if 'properties' in combined_df.columns:
        combined_df['properties'] = combined_df['properties'].apply(
            lambda x: str(x) if isinstance(x, dict) else x
        )
        
    return combined_df


def load_raw_microdata_node(file_paths: List[Path], questionnaire: pd.DataFrame) -> pd.DataFrame:
    """
    Loads raw microdata (answers) from extracted folders.
    Applies multi-question transformation using questionnaire metadata but does not
    merge questionnaire columns into the output. Values are normalized and stringified.
    """
    logger.info(f"Processing raw microdata for {len(file_paths)} paths")
    survey_info = get_survey_info(file_paths)

    dfs_microdata = []

    for survey_questionnaire, questionnaires_details in survey_info.items():
        for questionnaires_version, file_paths_detail in questionnaires_details.items():
            tabular_path = file_paths_detail.get('Tabular')

            if not tabular_path:
                logger.warning(
                    f"Skipping raw microdata load for {survey_questionnaire} v{questionnaires_version}: "
                    "missing Tabular export"
                )
                continue

            try:
                df_questionnaires = questionnaire[questionnaire['qnr'] == survey_questionnaire]
                df_microdata = get_microdata_raw(tabular_path, df_questionnaires)
                dfs_microdata.append(df_microdata)
                logger.info(f"Loaded raw microdata for {survey_questionnaire} v{questionnaires_version}")
            except Exception as e:
                logger.error(f"Failed to load raw microdata for {survey_questionnaire} v{questionnaires_version}. Skipping. Error: {str(e)}")
                continue

    if not dfs_microdata:
        return pd.DataFrame()

    combined_df = pd.concat(dfs_microdata)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df


def merge_microdata_questionnaire_node(raw_microdata: pd.DataFrame, questionnaire: pd.DataFrame) -> pd.DataFrame:
    """
    Merges raw microdata with questionnaire metadata and normalizes column names.
    """
    logger.info("Merging raw microdata with questionnaire metadata")
    merged = merge_microdata_questionnaire(raw_microdata, questionnaire)
    merged.reset_index(drop=True, inplace=True)
    return merged

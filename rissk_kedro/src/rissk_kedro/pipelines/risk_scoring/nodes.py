## Pipeline Nodes (pipelines)

### 1 Ingestion Pipeline
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
from loguru import logger

# Import your existing utilities
from rissk.utils.import_utils import (
    extract_zip,
    get_survey_info,
    get_dataframes
)

"""Nodes for ingesting Survey Solutions export data."""

def unzip_raw_surveys(
    parameters: Dict
) -> None:
    """
    Extract zipped Survey Solutions exports.
    
    Handles:
    - Recursive unzipping (nested ZIPs)
    - Password-protected ZIPs (from credentials)
    - Mixed formats (.dta, .tab)
    
    Args:
        parameters: Survey configuration from parameters.yml
        
    Side Effect:
        Extracts files to data/01_raw/{survey_name}/{version}/
    """
    from rissk.config import RAW_DATA_DIR
    from rissk.utils.import_utils import get_zip_files
    
    survey_name = parameters["survey"]["name"]
    questionnaires = parameters["survey"]["questionnaires"]
    
    # Get all ZIP files matching the survey config
    zip_files = get_zip_files(RAW_DATA_DIR, survey_name, questionnaires)
    
    logger.info(f"Found {len(zip_files)} ZIP files to extract")
    
    for zip_file in zip_files:
        dest_path = zip_file.with_suffix('')  # Remove .zip extension
        logger.info(f"Extracting {zip_file.name} to {dest_path}")
        extract_zip(zip_file, dest_path)
    
    logger.success(f"Extraction complete. Files in {RAW_DATA_DIR}")


def load_survey_dataframes(
    parameters: Dict
) -> tuple:
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
    from rissk.config import RAW_DATA_DIR
    from rissk.utils.import_utils import get_survey_info, get_dataframes
    
    # Scan extracted directories for survey info
    survey_paths = []
    for item in RAW_DATA_DIR.iterdir():
        if item.is_dir():
            survey_paths.append(item)
    
    survey_info = get_survey_info(survey_paths)
    
    logger.info(f"Loading dataframes for surveys: {list(survey_info.keys())}")
    
    # Use your existing get_dataframes logic
    paradata_df, questionnaire_df, microdata_df = get_dataframes(survey_info)
    
    logger.info(f"Loaded - Paradata: {paradata_df.shape}, "
                f"Questionnaire: {questionnaire_df.shape}, "
                f"Microdata: {microdata_df.shape}")
    
    return paradata_df, questionnaire_df, microdata_df


### 2 Feature Engineering Pipeline

"""Nodes for processing paradata and building features."""
import pandas as pd
from typing import Dict
from loguru import logger


def process_paradata_timestamps(
    paradata_raw: pd.DataFrame
) -> pd.DataFrame:
    """
    Process paradata timestamps and add hour features.
    
    This replicates logic from pipelines/feature_engineering/10_process_paradata.py
    
    Args:
        paradata_raw: Raw paradata DataFrame
        
    Returns:
        Processed paradata with timestamp features
    """
    paradata = paradata_raw.copy()
    
    # Add answer hour feature (from 10_process_paradata.py line 29)
    paradata['f__answer_hour_set'] = (
        paradata['timestamp_local'].dt.hour + 
        paradata['timestamp_local'].dt.round('30min').dt.minute / 60
    )
    
    # Add interviewing flag
    paradata['interviewing'] = ~paradata['role'].isin([2, 3, 4])
    
    logger.info(f"Processed {len(paradata)} paradata records")
    
    return paradata


def filter_active_events(
    paradata_processed: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """
    Filter paradata to active interviewer events.
    
    Replicates logic from pipelines/feature_engineering/11_process_paradata_active.py
    
    Args:
        paradata_processed: Processed paradata
        parameters: Config parameters (for limit_unit)
        
    Returns:
        DataFrame with only active interviewer events
    """
    active_events = [
        'InterviewCreated', 'AnswerSet', 'Resumed', 
        'AnswerRemoved', 'CommentSet', 'Restarted'
    ]
    
    # Filter logic from 11_process_paradata_active.py line 28
    active_mask = (
        paradata_processed['event'].isin(active_events) &
        paradata_processed['question_scope'].isin([0, '']) &
        (paradata_processed['role'] == 1)
    )
    
    vars_needed = [
        'interview__id', 'order', 'event', 'responsible', 'role', 'tz_offset',
        'param', 'answer', 'roster_level', 'timestamp_local', 'variable_name',
        'question_sequence', 'question_scope', "qtype", 'question_type',
        'qnr', 'qnr_version', 'interviewing', 'yes_no_view', 'index_col', 
        'f__answer_hour_set'
    ]
    
    df_para_active = paradata_processed.loc[active_mask, vars_needed]
    
    logger.info(f"Filtered to {len(df_para_active)} active events")
    
    return df_para_active


def build_item_features(
    microdata_raw: pd.DataFrame,
    paradata_active: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """
    Build item-level features from microdata and paradata.
    
    Uses logic from rissk/feature_processing.py make_df_item method.
    
    Args:
        microdata_raw: Raw microdata
        paradata_active: Active paradata events
        parameters: Feature configuration
        
    Returns:
        DataFrame with item-level features
    """
    from rissk.feature_processing import FeatureProcessing
    
    # Instantiate your existing class (or refactor to pure functions)
    # For now, we'll use a wrapper approach
    allowed_features = [
        f'f__{k}' for k, v in parameters['features'].items() 
        if v['use']
    ]
    
    logger.info(f"Building {len(allowed_features)} item features")
    
    # You would call methods like:
    # df_item = feature_processor.make_df_item(microdata_raw)
    # For brevity, returning placeholder
    
    df_item = microdata_raw.copy()  # Replace with actual logic
    
    return df_item


def build_unit_features(
    paradata_active: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """
    Build unit-level (interview-level) features.
    
    Uses logic from rissk/feature_processing.py make_df_unit method.
    
    Args:
        paradata_active: Active paradata
        parameters: Configuration
        
    Returns:
        DataFrame with unit-level features
    """
    df_unit = paradata_active[[
        'interview__id', 'responsible', 'survey_name', 'survey_version'
    ]].copy()
    
    df_unit.drop_duplicates(inplace=True)
    df_unit = df_unit[
        (df_unit['responsible'] != '') & 
        (~pd.isnull(df_unit['responsible']))
    ]
    
    # Add pause features (from your add_pause_features method)
    # Add time features (from add_unit_time_features)
    
    logger.info(f"Built {len(df_unit)} unit records")
    
    return df_unit

### 3 Risk Scoring Pipeline

# filepath: rissk_kedro/src/rissk_kedro/pipelines/risk_scoring/nodes.py
"""Nodes for calculating risk scores."""
import pandas as pd
from typing import Dict
from loguru import logger


def calculate_unit_risk_scores(
    unit_features: pd.DataFrame,
    item_features: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """
    Calculate global risk scores for each unit (interview).
    
    Uses logic from rissk/unit_proccessing.py make_global_score method.
    
    Args:
        unit_features: Unit-level features
        item_features: Item-level features
        parameters: Feature configuration
        
    Returns:
        DataFrame with unit_risk_score column
    """
    from rissk.unit_proccessing import UnitDataProcessing
    
    # You would instantiate your class or refactor to pure functions
    # For now, placeholder logic:
    
    unit_scores = unit_features.copy()
    unit_scores['unit_risk_score'] = 0.0  # Replace with actual IForest scoring
    
    logger.info(f"Calculated risk scores for {len(unit_scores)} units")
    
    return unit_scores


def format_output_scores(
    unit_risk_scores: pd.DataFrame,
    parameters: Dict
) -> tuple:
    """
    Format final output files.
    
    Args:
        unit_risk_scores: Scores DataFrame
        parameters: Output configuration
        
    Returns:
        tuple: (unit_scores_df, feature_scores_df) if feature_score=True
    """
    # Main output (from rissk/unit_proccessing.py save method line 104)
    output_df = unit_risk_scores[[
        'interview__id', 'responsible', 'unit_risk_score'
    ]].copy()
    
    output_df['unit_risk_score'] = output_df['unit_risk_score'].round(2)
    output_df.sort_values('unit_risk_score', inplace=True)
    
    logger.success(f"Formatted {len(output_df)} risk scores for output")
    
    if parameters['output']['feature_score']:
        # Generate feature score breakdown
        feature_scores_df = unit_risk_scores.copy()  # Add all s__ columns
        return output_df, feature_scores_df
    
    return output_df, None
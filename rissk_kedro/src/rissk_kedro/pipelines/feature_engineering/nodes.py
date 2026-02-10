"""Nodes for processing paradata and building features."""
import pandas as pd
import numpy as np
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
    
    # Mark interviewing events (before Supervisor/HQ interaction)
    events_split = ['RejectedBySupervisor', 'OpenedBySupervisor', 'OpenedByHQ', 'RejectedByHQ']
    paradata['flag'] = paradata['event'].isin(events_split)
    
    # Count flagged events for each interview
    paradata['cumulative_flag'] = paradata.groupby('interview__id')['flag'].cumsum()
    paradata['interviewing'] = np.where(paradata['cumulative_flag'] > 0, False, True)
    
    logger.info(f"Processed {len(paradata)} paradata records with timestamp features")
    
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
    
    # Filter to active events
    active_mask = (
        paradata_processed['event'].isin(active_events) &
        paradata_processed['interviewing']
    )
    
    # Apply limit_unit filter if specified
    limit_unit = parameters.get('processing', {}).get('limit_unit')
    if limit_unit is not None:
        active_mask = active_mask & (paradata_processed['interview__id'].isin(limit_unit))
    
    df_para_active = paradata_processed[active_mask].copy()
    
    logger.info(f"Filtered to {len(df_para_active)} active events")
    
    return df_para_active


def build_item_features(
    microdata_raw: pd.DataFrame,
    paradata_active: pd.DataFrame,
    questionnaire_raw: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """
    Build item-level features from microdata and paradata.
    
    Uses logic from pipelines/feature_engineering/12_process_items.py
    
    Args:
        microdata_raw: Raw microdata
        paradata_active: Active paradata events
        questionnaire_raw: Questionnaire structure
        parameters: Feature configuration
        
    Returns:
        DataFrame with item-level features
    """
    logger.info("Building item-level features")
    
    # Create index column for joining
    def make_index_col(df):
        mask = (~df[['interview__id', 'variable_name', 'roster_level']].isnull()) & \
                (df[['interview__id', 'variable_name', 'roster_level']] != '')
        filtered_df = df.where(mask, '')
        df['index_col'] = (
            filtered_df['interview__id'].astype(str) + '__' +
            filtered_df['variable_name'].astype(str) + '__' +
            filtered_df['roster_level'].astype(str)
        )
        return df
    
    microdata = make_index_col(microdata_raw.copy())
    
    # Select relevant columns
    item_level_columns = ['interview__id', 'variable_name', 'roster_level']
    df_item = microdata[['value', "qtype", 'is_integer', 'qnr_seq',
                         'n_answers', 'answer_sequence',
                         'cascade_from_question_id', 'is_filtered_combobox',
                         'index_col'] + item_level_columns].copy()
    
    # Merge with active paradata
    paradata_columns = ['responsible', 'f__answer_hour_set', 'interviewing', 'tz_offset']
    answer_set_mask = (paradata_active['event'] == 'AnswerSet')
    data = paradata_active[answer_set_mask].drop_duplicates(subset='index_col', keep='last')
    
    df_item = df_item.merge(
        data[paradata_columns + ['index_col']], 
        how='left',
        on='index_col'
    )
    
    # Keep only interviewing events
    df_item = df_item[df_item['interviewing'] == True]
    
    logger.info(f"Built {len(df_item)} item feature records")
    
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
    
    logger.info(f"Built {len(df_unit)} unit records")
    
    return df_unit

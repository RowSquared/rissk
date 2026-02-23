"""Nodes for processing paradata and building features."""
import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger


def process_paradata_node(
    paradata_interim: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """
    Process paradata timestamps, flags, and index creation.
    
    Args:
        paradata_interim: Interim paradata DataFrame
        parameters: Pipeline parameters
        
    Returns:
        Processed paradata DataFrame
    """
    paradata = paradata_interim.copy()
    
    # Calculate f__answer_hour_set
    paradata['f__answer_hour_set'] = (
        paradata['timestamp_local'].dt.hour + 
        paradata['timestamp_local'].dt.round('30min').dt.minute / 60
    )
    
    # Calculate interviewing flag
    events_split = ['RejectedBySupervisor', 'OpenedBySupervisor', 'OpenedByHQ', 'RejectedByHQ']
    paradata['flag'] = paradata['event'].isin(events_split)
    
    # Count flagged events for each interview
    paradata['cumulative_flag'] = paradata.groupby('interview__id')['flag'].cumsum()
    paradata['interviewing'] = np.where(paradata['cumulative_flag'] > 0, False, True)
    
    # Filter interviewing == True AND role == 1
    paradata.drop(['flag', 'cumulative_flag'], axis=1, inplace=True)
    paradata = paradata[(paradata['interviewing'] == True) & (paradata['role'] == 1)].copy()
    
    # Implement make_index_col logic (concat ID parts)
    # Using '_' separator to match previous notebook logic
    def make_index_col(df):
        mask = (~df[['interview__id', 'variable_name', 'roster_level']].isnull()) & \
                (df[['interview__id', 'variable_name', 'roster_level']] != '')
        filtered_df = df.where(mask, '')

        # Concatenate the columns with an underscore separator
        df['index_col'] = (
            filtered_df['interview__id'].astype(str) + "_" +
            filtered_df['variable_name'].astype(str) + "_" +
            filtered_df['roster_level'].astype(str)
        )
        df['index_col'] = df['index_col'].str.strip('_')
        return df
    
    paradata = make_index_col(paradata)
    
    # Sort by interview__id, order
    paradata.sort_values(['interview__id', 'order'], inplace=True)
    paradata.reset_index(drop=True, inplace=True)
    
    # Limit Unit Logic
    limit_unit = parameters.get('processing', {}).get('limit_unit')
    if limit_unit is not None:
        consent_variable = next(iter(limit_unit))
        consent_value = str(limit_unit[consent_variable])
        
        cond1 = (paradata['variable_name'] == consent_variable)
        cond2 = (paradata['answer'] == consent_value)
        
        filtered_interview_id = paradata[cond1 & cond2]['interview__id'].unique()
        paradata = paradata[paradata['interview__id'].isin(filtered_interview_id)].copy()
    
    return paradata


def filter_active_paradata_node(
    paradata_processed: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """
    Filter paradata to active events.
    
    Args:
        paradata_processed: Processed paradata DataFrame
        parameters: Pipeline parameters
        
    Returns:
        Active paradata DataFrame: keep active events, prior rejection/review events, for questions with scope interviewer
    """
    active_events = [
        'InterviewCreated', 'AnswerSet', 'Resumed', 
        'AnswerRemoved', 'CommentSet', 'Restarted'
    ]
    # only keep events done by interview (in most cases this should be all, after above filters,
    # just in case supervisor or HQ answered something while interviewer answered on web mode)
    # keep active events, prior rejection/review events, for questions with scope interviewer    

    # Filter conditions
    active_mask = (
        (paradata_processed['event'].isin(active_events)) &
        (paradata_processed['question_scope'].isin([0, ''])) &
        (paradata_processed['role'] == 1)
    )
    
    vars_needed = [
        'interview__id', 'order', 'event', 'responsible', 'role', 'tz_offset',
        'param', 'answer', 'roster_level', 'timestamp_local', 'variable_name',
        'question_sequence', 'question_scope', "qtype", 'question_type',
        'qnr', 'qnr_version', 'interviewing', 'yes_no_view', 'index_col', 'f__answer_hour_set'
    ]
    
    # Only keep columns present in the dataframe
    vars_needed = [col for col in vars_needed if col in paradata_processed.columns]
    
    df_para_active = paradata_processed.loc[active_mask, vars_needed].copy()
    
    return df_para_active


def build_item_features(
    microdata_raw: pd.DataFrame,
    paradata_active: pd.DataFrame,
    questionnaire_raw: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """
    Build item-level features from microdata and paradata.
    
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
    # Updated separator to '_' to match process_paradata_node
    def make_index_col(df):
        mask = (~df[['interview__id', 'variable_name', 'roster_level']].isnull()) & \
                (df[['interview__id', 'variable_name', 'roster_level']] != '')
        filtered_df = df.where(mask, '')
        df['index_col'] = (
            filtered_df['interview__id'].astype(str) + '_' +
            filtered_df['variable_name'].astype(str) + '_' +
            filtered_df['roster_level'].astype(str)
        )
        df['index_col'] = df['index_col'].str.strip('_')
        return df
    
    if microdata_raw.empty:
        logger.warning("Microdata is empty")
        return pd.DataFrame()

    microdata = make_index_col(microdata_raw.copy())
    
    # Select relevant columns
    item_level_columns = ['interview__id', 'variable_name', 'roster_level']
    
    # Identify available columns from the desired list
    desired_cols = ['value', "qtype", 'is_integer', 'qnr_seq',
                    'n_answers', 'answer_sequence',
                    'cascade_from_question_id', 'is_filtered_combobox',
                    'index_col'] + item_level_columns
                    
    available_cols = [c for c in desired_cols if c in microdata.columns]
    
    df_item = microdata[available_cols].copy()
    
    # Merge with active paradata
    paradata_columns = ['responsible', 'f__answer_hour_set', 'interviewing', 'tz_offset']
    answer_set_mask = (paradata_active['event'] == 'AnswerSet')
    data = paradata_active[answer_set_mask].drop_duplicates(subset='index_col', keep='last')
    
    # Filter paradata columns to those present in data
    available_para_cols = [col for col in paradata_columns if col in data.columns]
    
    df_item = df_item.merge(
        data[available_para_cols + ['index_col']], 
        how='left',
        on='index_col'
    )
    
    # Keep only interviewing events if column exists
    if 'interviewing' in df_item.columns:
        df_item = df_item[df_item['interviewing'] == True]
    
    logger.info(f"Built {len(df_item)} item feature records")
    
    return df_item


def build_unit_features(
    paradata_active: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """
    Build unit-level (interview-level) features.
    
    Args:
        paradata_active: Active paradata
        parameters: Configuration
        
    Returns:
        DataFrame with unit-level features
    """
    # Use qnr/qnr_version as survey_name/survey_version
    cols_map = {
        'interview__id': 'interview__id',
        'responsible': 'responsible',
        'qnr': 'survey_name',
        'qnr_version': 'survey_version'
    }
    
    # Only select columns that exist
    available_cols = [c for c in cols_map.keys() if c in paradata_active.columns]
    
    df_unit = paradata_active[available_cols].copy()
    
    # Rename columns to match expected output
    df_unit.rename(columns=cols_map, inplace=True)
    
    df_unit.drop_duplicates(inplace=True)
    
    if 'responsible' in df_unit.columns:
        df_unit = df_unit[
            (df_unit['responsible'] != '') & 
            (~pd.isnull(df_unit['responsible']))
        ]
    
    logger.info(f"Built {len(df_unit)} unit records")
    
    return df_unit


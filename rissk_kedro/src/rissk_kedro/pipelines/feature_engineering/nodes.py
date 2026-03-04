"""Nodes for processing paradata and building features."""
import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger

from rissk.feature_processing_kedro import make_index_col


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
    
    # Use shared helper to avoid drift with feature_processing_kedro
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
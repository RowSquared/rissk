import pandas as pd
from typing import Dict, Any, Tuple
import logging

from rissk.item_processing_kedro import (
    calculate_answer_hour_set_score,
    calculate_sequence_jump_score,
    calculate_first_decimal_score,
    calculate_answer_changed_score,
    calculate_answer_removed_score,
    calculate_answer_position_score,
    calculate_answer_selected_score,
    calculate_answer_duration_score,
    calculate_single_question_score,
    calculate_multi_option_question_score,
    calculate_first_digit_score,
    calculate_gps_score
)
from rissk.unit_processing_kedro import (
    calculate_global_score,
    aggregate_unit_score_mean,
    aggregate_item_to_unit_scores,
    calculate_unit_level_scores,
    aggregate_item_to_responsible_scores,
    calculate_responsible_score
)

logger = logging.getLogger(__name__)

def calculate_item_scores(df_item: pd.DataFrame, df_item_removed: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Run item level scoring applying various mathematical models.
    """
    logger.info("Calculating Item Scores...")
    df_scored = calculate_answer_hour_set_score(df_item, parameters)
    df_scored = calculate_sequence_jump_score(df_scored, parameters)
    df_scored = calculate_first_decimal_score(df_scored, parameters)
    df_scored = calculate_answer_changed_score(df_scored, parameters)
    df_scored = calculate_answer_position_score(df_scored, parameters)
    df_scored = calculate_answer_selected_score(df_scored, parameters)
    df_scored = calculate_answer_duration_score(df_scored, parameters)
    df_scored = calculate_single_question_score(df_scored, parameters)
    df_scored = calculate_multi_option_question_score(df_scored, parameters)
    df_scored = calculate_first_digit_score(df_scored, parameters)
    df_scored = calculate_gps_score(df_scored, parameters)
    
    # Needs to handle distinct output of removed scores mapping
    df_removed_scored = calculate_answer_removed_score(df_scored, df_item_removed, parameters)
    
    # ... other item scores would be chained here ...
    return df_scored

def calculate_unit_scores(df_unit: pd.DataFrame, df_item_scores: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate item scores to unit, extract responsible scores, and calculate global risk.
    """
    logger.info("Calculating Unit Scores and Global Risk...")
    
    # 1. Aggregate item-level scores up to unit level
    df_unit_scored = aggregate_item_to_unit_scores(df_unit, df_item_scores)
    
    # 2. Add pure unit-level calculations
    df_unit_scored = calculate_unit_level_scores(df_unit_scored, parameters)

    # 3. Aggregate item-level scores up to responsible level
    df_resp_scored = pd.DataFrame()
    df_resp_scored = aggregate_item_to_responsible_scores(df_resp_scored, df_item_scores)
    
    # 4. Calculate final responsible score via PCA
    restricted_columns = parameters.get('scoring', {}).get('restricted_columns', [])
    df_resp_scored = calculate_responsible_score(df_resp_scored, restricted_columns)
    
    # Determine all scored columns dynamically (s_*)
    score_columns = [col for col in df_unit_scored.columns if col.startswith('s__')]
    
    # 5. Calculate final global unit risk score
    df_final_unit = calculate_global_score(
        df_unit_scores=df_unit_scored, 
        df_resp_scores=df_resp_scored, 
        score_columns=score_columns,
        combine_resp_score=True,
        restricted_columns=restricted_columns
    )
    
    return df_final_unit, df_resp_scored

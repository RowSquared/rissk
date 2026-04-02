import pandas as pd
from typing import Dict, Any, Tuple
import logging

from rissk.item_processing_kedro import (
    calculate_answer_hour_set_score,
    calculate_sequence_jump_score,
    calculate_first_decimal_score,
    calculate_answer_changed_score,
    # calculate_answer_removed_score is intentionally absent: s__answer_removed is
    # computed at unit level from the removed_answers dataset by calculate_answer_removed_score_from_df
    # so that AnswerRemoved events for items deleted from microdata are not missed.
    calculate_answer_removed_score_from_df,
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

def calculate_item_scores(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Run item level scoring applying various mathematical models.
    f__answer_removed is already present in df_item from the feature creation pipeline.
    Each scoring function is only executed when its corresponding feature has use: true
    in parameters['features'], matching the feature creation pipeline behaviour.
    """
    questionnaires = parameters.get('survey', {}).get('questionnaires', [])
    lines = [
        "=" * 55,
        "  RISSK SCORING",
        "=" * 55,
        "  Questionnaires:",
    ]
    for q in questionnaires:
        lines.append(f"    • {q['name']}")
    lines.append("=" * 55)
    logger.info("\n" + "\n".join(lines))
    logger.info("Calculating Item Scores...")
    features = parameters.get('features', {})
    df_scored = df_item

    if features.get('answer_hour_set', {}).get('use', False):
        logger.info("Calculating answer_hour_set_score")
        df_scored = calculate_answer_hour_set_score(df_scored, parameters)

    if features.get('sequence_jump', {}).get('use', False):
        logger.info("Calculating sequence_jump_score")
        df_scored = calculate_sequence_jump_score(df_scored, parameters)

    if features.get('first_decimal', {}).get('use', False):
        logger.info("Calculating first_decimal_score")
        df_scored = calculate_first_decimal_score(df_scored, parameters)

    if features.get('answer_changed', {}).get('use', False):
        logger.info("Calculating answer_changed_score")
        df_scored = calculate_answer_changed_score(df_scored, parameters)

    # s__answer_removed is not computed here — see calculate_answer_removed_score_from_df
    # in calculate_unit_scores, which scores from the removed_answers dataset to match legacy coverage.

    if features.get('answer_position', {}).get('use', False):
        logger.info("Calculating answer_position_score")
        df_scored = calculate_answer_position_score(df_scored, parameters)

    if features.get('answer_selected', {}).get('use', False):
        logger.info("Calculating answer_selected_score")
        df_scored = calculate_answer_selected_score(df_scored, parameters)

    if features.get('answer_duration', {}).get('use', False):
        logger.info("Calculating answer_duration_score")
        df_scored = calculate_answer_duration_score(df_scored, parameters)

    if features.get('single_question', {}).get('use', False):
        logger.info("Calculating single_question_score")
        df_scored = calculate_single_question_score(df_scored)

    if features.get('multi_option_question', {}).get('use', False):
        logger.info("Calculating multi_option_question_score")
        df_scored = calculate_multi_option_question_score(df_scored)

    if features.get('first_digit', {}).get('use', False):
        logger.info("Calculating first_digit_score")
        df_scored = calculate_first_digit_score(df_scored)

    if features.get('gps', {}).get('use', False):
        logger.info("Calculating gps_score")
        df_scored = calculate_gps_score(df_scored, parameters)

    return df_scored

def calculate_unit_scores(
        df_unit: pd.DataFrame, 
        df_item_scores: pd.DataFrame, 
        parameters: Dict[str, Any], removed_answers: pd.DataFrame = None
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate item scores to unit, extract responsible scores, and calculate global risk.

    removed_answers is the pre-aggregated AnswerRemoved dataset produced by build_removed_answers_node.
    It is used to compute s__answer_removed at unit level, matching legacy behaviour where
    items deleted from microdata (absent from df_item) are still counted.
    """
    logger.info("Calculating Unit Scores and Global Risk...")
    features = parameters.get('features', {})

    # 1. Aggregate item-level scores up to unit level.
    # s__answer_removed is excluded from this aggregation (see aggregate_item_to_unit_scores);
    # it is handled below using paradata_full to match legacy coverage.
    df_unit_scored = aggregate_item_to_unit_scores(df_unit, df_item_scores)

    # 2a. Score answer_removed at unit level from paradata_full.
    # This replicates legacy make_score_unit__answer_removed which read from df_paradata
    # directly and therefore included AnswerRemoved events for items later deleted from
    # microdata. Falling back to the df_item-based mean when paradata_full is unavailable.
    if features.get('answer_removed', {}).get('use', False):
        if removed_answers is not None and not removed_answers.empty:
            unit_removed = calculate_answer_removed_score_from_df(removed_answers, parameters)
            df_unit_scored['s__answer_removed'] = df_unit_scored['interview__id'].map(unit_removed).fillna(0)
        elif 's__answer_removed' in df_item_scores.columns:
            logger.warning(
                "removed_answers not available; falling back to df_item-based s__answer_removed "
                "aggregation (may undercount removals for deleted items)."
            )
            data = df_item_scores.groupby('interview__id')['s__answer_removed'].mean()
            df_unit_scored['s__answer_removed'] = df_unit_scored['interview__id'].map(data).fillna(0)
    
    # 2b. Add pure unit-level calculations
    df_unit_scored = calculate_unit_level_scores(df_unit_scored, parameters)

    # 3. Aggregate item-level scores up to responsible level
    df_resp_scored = pd.DataFrame()
    df_resp_scored = aggregate_item_to_responsible_scores(df_resp_scored, df_item_scores)
    
    # 4. Calculate final responsible score via PCA
    restricted_columns = parameters.get('unit_scoring', {}).get('restricted_columns', [])
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

    # 6. Merge responsible-level s__ columns back onto unit output.
    # Legacy save() merges _df_resp (which holds s__single_question,
    # s__multi_option_question, s__answer_position, s__first_digit) back
    # onto _df_unit by responsible so those scores appear in the feature CSV.
    resp_s_cols = [c for c in df_resp_scored.columns if c.startswith('s__')]
    if resp_s_cols and 'responsible' in df_resp_scored.columns and not df_resp_scored.empty:
        # Only bring in columns not already present at unit level
        new_resp_cols = [c for c in resp_s_cols if c not in df_final_unit.columns]
        if new_resp_cols:
            df_final_unit = df_final_unit.merge(
                df_resp_scored[['responsible'] + new_resp_cols],
                on='responsible',
                how='left'
            )

    return df_final_unit, df_resp_scored

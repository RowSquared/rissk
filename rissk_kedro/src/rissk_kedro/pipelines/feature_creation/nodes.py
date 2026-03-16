"""Nodes for the Feature Creation pipeline."""
import logging
import pandas as pd
from typing import Dict, Any

# Assuming rissk is importable as a package
# If running kedro from rissk_kedro root, ensure PYTHONPATH includes ../rissk
from rissk.feature_processing_kedro import (
    create_base_item_table,
    create_base_unit_table,
    enrich_item_features,
    enrich_unit_features,
    feat_answer_removed,
)

logger = logging.getLogger(__name__)

def create_base_item_table_node(
    microdata: pd.DataFrame,
    paradata_full: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Node wrapper for create_base_item_table."""
    return create_base_item_table(microdata, paradata_full, parameters)

def create_base_unit_table_node(
    paradata_full: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Node wrapper for create_base_unit_table."""
    return create_base_unit_table(paradata_full, parameters)

def enrich_item_features_node(
    item_features_base: pd.DataFrame,
    paradata_full: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Node wrapper for enrich_item_features.
    paradata_full: all processed events, role=1, interviewing=True (equivalent to self.df_paradata).
    """
    return enrich_item_features(item_features_base, paradata_full, parameters)

def enrich_unit_features_node(
    unit_features_base: pd.DataFrame,
    item_features: pd.DataFrame,
    paradata_full: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Node wrapper for enrich_unit_features.
    paradata_full: all processed events, role=1, interviewing=True (equivalent to self.df_paradata).
    """
    return enrich_unit_features(unit_features_base, item_features, paradata_full, parameters)

def build_removed_answers_node(
    paradata_full: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Extract and aggregate all AnswerRemoved events as a standalone dataset.

    This node produces the removed_answers parquet which captures AnswerRemoved
    events for items that may no longer exist in microdata (deleted items), matching
    legacy get_feature_item__answer_removed behaviour. The output is consumed by
    the rissk_scoring pipeline to score s__answer_removed at unit level.
    """
    return feat_answer_removed(paradata_full)

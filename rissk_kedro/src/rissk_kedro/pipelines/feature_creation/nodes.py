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
)

logger = logging.getLogger(__name__)

def create_base_item_table_node(
    microdata: pd.DataFrame, 
    paradata_active: pd.DataFrame, 
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Node wrapper for create_base_item_table.
    """
    return create_base_item_table(microdata, paradata_active, parameters)

def create_base_unit_table_node(
    paradata_active: pd.DataFrame, 
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Node wrapper for create_base_unit_table.
    """
    return create_base_unit_table(paradata_active, parameters)

def enrich_item_features_node(
    item_features_base: pd.DataFrame,
    paradata_active: pd.DataFrame,
    paradata_full: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Node wrapper for enrich_item_features.
    paradata_active: active interviewer events only (equivalent to self.df_active_paradata).
    paradata_full: all processed events, role=1, interviewing=True (equivalent to self.df_paradata).
    """
    return enrich_item_features(item_features_base, paradata_active, paradata_full, parameters)

def enrich_unit_features_node(
    unit_features_base: pd.DataFrame,
    item_features: pd.DataFrame,
    paradata_full: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Node wrapper for enrich_unit_features.
    paradata_full: all processed events, role=1, interviewing=True (equivalent to self.df_paradata).
    """
    return enrich_unit_features(unit_features_base, item_features, paradata_full, parameters)

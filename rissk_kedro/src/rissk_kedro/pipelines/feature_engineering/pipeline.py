"""Feature engineering pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    process_paradata_node,
    filter_active_paradata_node,
    build_item_features,
    build_unit_features
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature engineering pipeline.
    
    Returns:
        A pipeline that processes paradata and builds features.
    """
    return pipeline([
        node(
            func=process_paradata_node,
            inputs=["paradata_interim", "parameters"],
            outputs="paradata_processed",
            name="process_paradata_node",
        ),
        node(
            func=filter_active_paradata_node,
            inputs=["paradata_processed", "parameters"],
            outputs="paradata_active",
            name="filter_active_paradata_node",
        ),
        node(
            func=build_item_features,
            inputs=["raw_microdata", "paradata_active", "raw_questionnaire", "parameters"],
            outputs="item_features",
            name="build_item_features_node",
        ),
        node(
            func=build_unit_features,
            inputs=["paradata_active", "parameters"],
            outputs="unit_features",
            name="build_unit_features_node",
        ),
    ])

"""Feature engineering pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    process_paradata_timestamps,
    filter_active_events,
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
            func=process_paradata_timestamps,
            inputs="paradata_raw",
            outputs="paradata_processed",
            name="process_timestamps_node",
        ),
        node(
            func=filter_active_events,
            inputs=["paradata_processed", "parameters"],
            outputs="paradata_active",
            name="filter_active_events_node",
        ),
        node(
            func=build_item_features,
            inputs=["microdata_raw", "paradata_active", "questionnaire_raw", "parameters"],
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

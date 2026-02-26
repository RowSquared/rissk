"""Feature engineering pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    process_paradata_node,
    filter_active_paradata_node,
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
    ])

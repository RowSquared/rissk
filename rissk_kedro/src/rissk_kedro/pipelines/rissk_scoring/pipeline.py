"""Rissk scoring pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import calculate_item_scores, calculate_unit_scores

def create_pipeline(**kwargs) -> Pipeline:
    """Create the scoring pipeline.
    
    Returns:
        A pipeline that calculates item and unit risk scores.
    """
    return pipeline([
        node(
            func=calculate_item_scores,
            inputs=["item_features", "parameters"],
            outputs="item_scores",
            name="calculate_item_scores_node",
        ),
        node(
            func=calculate_unit_scores,
            inputs=["unit_features", "item_scores", "parameters"],
            outputs=["unit_risk_scores", "responsible_scores"],
            name="calculate_unit_scores_node",
        ),
    ])

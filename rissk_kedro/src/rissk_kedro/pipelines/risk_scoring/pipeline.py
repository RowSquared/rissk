"""Risk scoring pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import calculate_unit_risk_scores, format_output_scores


def create_pipeline(**kwargs) -> Pipeline:
    """Create the risk scoring pipeline.
    
    Returns:
        A pipeline that calculates unit risk scores.
    """
    return pipeline([
        node(
            func=calculate_unit_risk_scores,
            inputs=["unit_features", "item_features", "parameters"],
            outputs="unit_risk_scores_raw",
            name="calculate_scores_node",
        ),
        node(
            func=format_output_scores,
            inputs=["unit_risk_scores_raw", "parameters"],
            outputs=["unit_risk_scores", "unit_feature_scores"],
            name="format_outputs_node",
        ),
    ])
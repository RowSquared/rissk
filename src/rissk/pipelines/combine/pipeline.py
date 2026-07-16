"""Combine pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import combine_microdata_node


def create_pipeline(**kwargs) -> Pipeline:
    """Union the per-questionnaire microdata into the survey-level file.

    Runs once, after all per-questionnaire runs of __default__.
    """
    return pipeline([
        node(
            func=combine_microdata_node,
            inputs="microdata_by_qnr",
            outputs="microdata_combined",
            name="combine_microdata_node",
        ),
    ])

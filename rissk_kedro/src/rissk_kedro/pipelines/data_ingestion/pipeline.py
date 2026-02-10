"""Data ingestion pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import unzip_raw_surveys, load_survey_dataframes


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data ingestion pipeline.
    
    Returns:
        A pipeline that extracts and loads Survey Solutions data.
    """
    return pipeline([
        node(
            func=unzip_raw_surveys,
            inputs=["raw_zip_files", "params:survey"],
            outputs=None,  # Side effect: extracts to same directory
            name="unzip_surveys_node",
        ),
        node(
            func=load_survey_dataframes,
            inputs="params:survey",
            outputs=["paradata_raw", "questionnaire_raw", "microdata_raw"],
            name="load_dataframes_node",
        ),
    ])

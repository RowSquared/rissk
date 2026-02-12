from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    extract_zip_files_node,
    filter_extracted_survey_paths_node,
    load_paradata_node, 
    load_questionnaire_node, 
    load_microdata_node
)
# catalog for path
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=extract_zip_files_node,
            inputs=[
                "params:ingestion.raw_data_path",
                "params:zip_password"
            ],
            outputs=None,
            name="extract_zip_files_node"
        ),
        node(
            func=filter_extracted_survey_paths_node,
            inputs=[
                "params:ingestion.raw_data_path",
                "params:survey.questionnaires",
            ],
            outputs="file_paths",
            name="filter_extracted_survey_paths_node"
        ),
        node(
            func=load_paradata_node,
            inputs="file_paths",
            outputs="paradata_interim",
            name="load_paradata_node"
        ),
        node(
            func=load_questionnaire_node,
            inputs="file_paths",
            outputs="raw_questionnaire",
            name="load_questionnaire_node"
        ),
        node(
            func=load_microdata_node,
            inputs="file_paths",
            outputs="raw_microdata",
            name="load_microdata_node"
        )
    ])

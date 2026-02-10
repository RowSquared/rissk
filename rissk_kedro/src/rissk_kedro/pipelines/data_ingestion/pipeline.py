from kedro.pipeline import Pipeline, node, pipeline
from .nodes import unzip_survey_data_node, load_survey_data_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=unzip_survey_data_node,
            inputs=[
                "params:survey.name",
                "params:ingestion.raw_data_path",
                "params:survey.questionnaires",
                "params:zip_password"
            ],
            outputs="extracted_survey_paths",
            name="unzip_survey_data_node"
        ),
        node(
            func=load_survey_data_node,
            inputs="extracted_survey_paths",
            outputs=["paradata_interim", "raw_questionnaire", "raw_microdata"],
            name="load_survey_data_node"
        )
    ])

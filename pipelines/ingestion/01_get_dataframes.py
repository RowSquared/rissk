# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: rissk
#     language: python
#     name: python3
# ---

# %% tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None
product = None
# %% [markdown]
# # Get Paradata, Microdata and Questionaire

# %% tags=[]
from rissk.config import SURVEY, QUESTIONAIRE, RAW_DATA_DIR
from rissk.utils.import_utils import get_zip_files, extract_zip, get_survey_info, get_dataframes


# %% [markdown]
# ## Extract Zip file

# %%
zip_files = get_zip_files(RAW_DATA_DIR, SURVEY, QUESTIONAIRE)

survey_paths = []
for zip_file in zip_files:
    project_path = zip_file.with_suffix('')
    survey_paths.append(project_path)
    extract_zip(zip_file, project_path)

# %% [markdown]
# ## Map Questionaire to paths

# %%

survey_info = get_survey_info(survey_paths)


# %% [markdown]
# ## Get Dataframes

# %%
dfs_questionnaires, dfs_microdata = get_dataframes(survey_info)
#dfs_paradata
# %% [markdown]
# ## Save Dataframes

# %%

#paradata_file = product['paradata']
questionnaire_file = product['questionnaire']
microdata_file = product['microdata']



#with open(paradata_file, 'wb') as f:
#    if 'answer_sequence' in dfs_paradata.columns:
#        dfs_paradata['answer_sequence'] = dfs_paradata['answer_sequence'].apply(str)
#    dfs_paradata.to_parquet(f)

with open(questionnaire_file, 'wb') as f:
    if 'answer_sequence' in dfs_questionnaires.columns:
        dfs_questionnaires['answer_sequence'] = dfs_questionnaires['answer_sequence'].apply(str)
    dfs_questionnaires.to_parquet(f)

with open(microdata_file, 'wb') as f:
    if 'answer_sequence' in dfs_microdata.columns:
        dfs_microdata['answer_sequence'] = dfs_microdata['answer_sequence'].apply(str)
    dfs_microdata.to_parquet(f)

# %%

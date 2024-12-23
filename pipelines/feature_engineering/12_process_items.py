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
# ---

# %% tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = ['01_get_dataframes', '10_process_paradata', '11_process_paradata_active']
product = None
limit_unit = None

# %%
import pandas as pd 
import numpy as np

# %%
microdata = pd.read_parquet(upstream['01_get_dataframes']['microdata'])
df_active_paradata = pd.read_parquet(upstream['11_process_paradata_active']['paradata_active'])

# %%
allowed_features = ['f__' + k for k, v in config['features'].items() if v['use']]
item_level_columns = ['interview__id', 'variable_name', 'roster_level']
df_paradata = self.process_paradata(paradata)
print('Paradata Processed')
_df_item = self.make_df_item(microdata)

# %% [markdown]
# ### Make df_item

# %%
microdata = make_index_col(microdata)
df_item = microdata[['value', "qtype", 'is_integer', 'qnr_seq',
                        'n_answers', 'answer_sequence',
                        'cascade_from_question_id', 'is_filtered_combobox',
                        'index_col'] + item_level_columns]

paradata_columns = ['responsible', 'f__answer_hour_set', 'interviewing', 'tz_offset']
# merge microdata with active pardata and keep only the last answer set
answer_set_mask = (df_active_paradata['event'] == 'AnswerSet')
data = df_active_paradata[answer_set_mask].drop_duplicates(subset='index_col', keep='last')
df_item = df_item.merge(data[paradata_columns + ['index_col']], how='left',
                        on='index_col')
# Remove items that are not in interviewing
df_item = df_item[df_item['interviewing'] == True]
df_item = add_sequence_features(df_item)

df_item = .add_item_time_features(df_item)


# %%
paradata_active_file = product['paradata_active']
with open(paradata_active_file, 'wb') as f:
    df_para_active.to_parquet(f)

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
upstream = ['get_dataframes']
product = None
limit_unit = None

# %%
import pandas as pd 
import numpy as np

# %%
paradata = pd.read_parquet(upstream['get_dataframes']['paradata'])
#paradata.fillna('', inplace=True)

paradata['f__answer_hour_set'] = (paradata['timestamp_local'].dt.hour + paradata['timestamp_local'].dt.round('30min').dt.minute / 60)

# interviewing, True prior to Supervisor/HQ interaction, else False
events_split = ['RejectedBySupervisor', 'OpenedBySupervisor', 'OpenedByHQ', 'RejectedByHQ']

# Create a flag indicating whether each row has an event in `events_split`
paradata['flag'] = paradata['event'].isin(events_split)

# %%
# Use `groupby` and `cumsum` to count how many flagged events occur for each group
# If the count is greater than 0, then the 'interviewing' column should be False
paradata['cumulative_flag'] = paradata.groupby('interview__id')['flag'].cumsum()
paradata['interviewing'] = np.where(paradata['cumulative_flag'] > 0, False, True)


# %%
def make_index_col(df):

    # Filter out columns with NaN and empty strings
    mask = (~df[['interview__id', 'variable_name', 'roster_level']].isnull()) & \
            (df[['interview__id', 'variable_name', 'roster_level']] != '')

    # Use the mask to replace invalid values with an empty string
    filtered_df = df.where(mask, '')

    # Concatenate the columns with an underscore separator
    df['index_col'] = filtered_df['interview__id'].astype(str) + "_" + \
                        filtered_df['variable_name'].astype(str) + "_" + \
                        filtered_df['roster_level'].astype(str)

    # Remove trailing and leading underscores if they exist
    df['index_col'] = df['index_col'].str.strip('_')
    return df.copy()


# %%
# Cleanup the intermediate columns
paradata.drop(['flag', 'cumulative_flag'], axis=1, inplace=True)
paradata = paradata[(paradata['interviewing'] == True) & (paradata['role'] == 1)].copy()

paradata = make_index_col(paradata)
paradata.sort_values(['interview__id', 'order'], inplace=True)
paradata.reset_index(inplace=True)

# %%
# if limit_unit is not None:
#     consent_variable = next(iter(limit_unit))  # Get the first (and only) key in the dictionary
#     # Careful! Answer value is a string in paradata.
#     # Therefore also consent_value must be set to a string.
#     consent_value = str(imit_unit[consent_variable])

#     cond1 = (paradata['variable_name'] == consent_variable)
#     cond2 = (paradata['answer'] == consent_value)

#     filtered_interview_id = paradata[cond1 & cond2]['interview__id'].unique()

#     paradata = paradata[paradata['interview__id'].isin(filtered_interview_id)].copy()

# %%
paradata_file = product['paradata']
with open(paradata_file, 'wb') as f:
    if 'answer_sequence' in paradata.columns:
        paradata['answer_sequence'] = paradata['answer_sequence'].apply(str)
    paradata.to_parquet(f)

# %%
# df_para_active, active events, prior rejection/review events, for questions with scope interviewer

active_events = ['InterviewCreated', 'AnswerSet', 'Resumed', 'AnswerRemoved', 'CommentSet', 'Restarted']
# only keep events done by interview (in most cases this should be all, after above filters,
# just in case supervisor or HQ answered something while interviewer answered on web mode)
# keep active events, prior rejection/review events, for questions with scope interviewer
active_mask = (paradata['event'].isin(active_events)) & \
                (paradata['question_scope'].isin([0, ''])) & \
                (paradata['role'] == 1)

vars_needed = ['interview__id', 'order', 'event', 'responsible', 'role', 'tz_offset',
                'param', 'answer', 'roster_level', 'timestamp_local', 'variable_name',
                'question_sequence', 'question_scope', "qtype", 'question_type',
                'survey_questionaire', 'questionaire_version', 'interviewing', 'yes_no_view', 'index_col', 'f__answer_hour_set'
                ]

df_para_active = paradata.loc[active_mask, vars_needed]

# %%
paradata_active_file = product['paradata_active']
with open(paradata_active_file, 'wb') as f:
    df_para_active.to_parquet(f)

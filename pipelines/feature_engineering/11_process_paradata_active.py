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
upstream = ['10_process_paradata']
product = None
limit_unit = None

# %%
import pandas as pd 
import numpy as np

# %%
paradata = pd.read_parquet(upstream['10_process_paradata']['paradata'])
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
                'qnr', 'qnr_version', 'interviewing', 'yes_no_view', 'index_col', 'f__answer_hour_set'
                ]

df_para_active = paradata.loc[active_mask, vars_needed]

# %%
paradata_active_file = product['paradata_active']
with open(paradata_active_file, 'wb') as f:
    df_para_active.to_parquet(f)

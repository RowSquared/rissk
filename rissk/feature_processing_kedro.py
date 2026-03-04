import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def make_index_col(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a unique index column based on interview_id, variable_name, and roster_level."""
    # Filter out columns with NaN and empty strings for the mask
    # Using fillna('') to handle NaNs safely for string concatenation
    
    # Create mask for valid rows (not null and not empty string in key columns)
    # Note: In Py3.13/Pandas 2.x, strict comparison rules apply.
    
    df_temp = df[['interview__id', 'variable_name', 'roster_level']].fillna('').astype(str)
    
    # Concatenate columns
    df['index_col'] = df_temp['interview__id'] + "_" + df_temp['variable_name'] + "_" + df_temp['roster_level']
    
    # Remove trailing and leading underscores
    df['index_col'] = df['index_col'].str.strip('_')
    return df

def get_numeric_mask(df_item: pd.DataFrame) -> pd.Series:
    """Returns a boolean mask for valid numeric question rows, matching the legacy numeric_question_mask."""
    return (
        (df_item["qtype"] == 'NumericQuestion') &
        (df_item['value'] != '') &
        (~pd.isnull(df_item['value'])) &
        (df_item['value'] != -999999999)
    )

def get_df_time(df_active_paradata: pd.DataFrame) -> pd.DataFrame:
    """Calculates time differences and durations from paradata."""
    df_time = df_active_paradata.copy()

    # Sort to ensure diff works correctly
    df_time = df_time.sort_values(['interview__id', 'timestamp_local'])

    # calculate time difference in seconds
    df_time['time_difference'] = df_time.groupby('interview__id')['timestamp_local'].diff()
    df_time['time_difference'] = df_time['time_difference'].dt.total_seconds()
    
    # Logic for f__time_changed (negative time diffs < -180s)
    df_time['f__time_changed'] = np.where(df_time['time_difference'] < -180, df_time['time_difference'], np.nan)
    
    # Mask negative time differences for duration calculations
    # Using pd.NA for nullable integers/floats in pandas if column allows, or np.nan
    df_time.loc[df_time['time_difference'] < 0, 'time_difference'] = np.nan

    # time for answers/comments
    df_time['f__answer_duration'] = df_time.loc[
        df_time['event'].isin(['AnswerSet', 'AnswerRemoved']), 'time_difference']
    df_time['f__comment_duration'] = df_time.loc[df_time['event'] == 'CommentSet', 'time_difference']
    df_time['f__pause_duration'] = df_time.loc[df_time['event'].isin(['Resumed', 'Restarted']), 'time_difference']

    # UNIT features helper logic
    active_events = ['AnswerSet', 'AnswerRemoved', 'CommentSet', 'Resumed', 'Restarted']
    
    # Calculate total duration (capped at 30 mins per event)
    condition = (df_time['event'].isin(active_events)) & (df_time['time_difference'] < 30 * 60)
    df_time['f__total_duration'] = df_time.loc[condition, 'time_difference']

    # Starting timestamp per interview: min timestamp of the first AnswerSet event per interview.
    # Using map on a pre-computed groupby result (matching the legacy approach).
    start_time_map = df_time[df_time['event'] == 'AnswerSet'].groupby('interview__id')['timestamp_local'].min()
    df_time['f__starting_timestamp'] = df_time['interview__id'].map(start_time_map)
    
    min_date = df_time['f__starting_timestamp'].min()
    if pd.notna(min_date):
        df_time['f__days_from_start'] = (df_time['timestamp_local'] - min_date).dt.days.abs()
    else:
        df_time['f__days_from_start'] = np.nan

    return df_time

def get_df_sequence(df_active_paradata: pd.DataFrame) -> pd.DataFrame:
    """Calculates sequence-based features (jumps, previous answers)."""
    # Filter for AnswerSet and get the last entry per index_col
    mask = df_active_paradata['event'] == 'AnswerSet'
    df_last = df_active_paradata[mask].groupby('index_col').last()

    # The groupby puts index_col in the index.
    # We need to sort by interview_id and order to reconstruct the sequence flow.
    # 'order' column is assumed to exist from ingestion.
    df_last = df_last.sort_values(['interview__id', 'order']).reset_index()

    # f__previous_question, f__previous_answer, f__previous_roster
    # Using shift on the group
    df_last['f__previous_question'] = df_last.groupby('interview__id')['variable_name'].shift()
    df_last['f__previous_answer'] = df_last.groupby('interview__id')['answer'].shift().fillna('')
    df_last['f__previous_roster'] = df_last.groupby('interview__id')['roster_level'].shift().fillna('')
    
    # f__sequence_jump
    # Calculate answer sequence (1, 2, 3...) based on actual occurrence
    df_last['answer_sequence'] = df_last.groupby('interview__id').cumcount() + 1
    
    # Diff between questionnaire sequence and answer sequence
    # Ensure types are compatible
    df_last['question_sequence'] = pd.to_numeric(df_last['question_sequence'], errors='coerce').fillna(0)
    df_last['diff'] = df_last['question_sequence'] - df_last['answer_sequence']
    
    # The 'jump' is the difference of the difference
    df_last['f__sequence_jump'] = df_last.groupby('interview__id')['diff'].diff()

    return df_last

def add_sequence_features(df_item: pd.DataFrame, df_sequence: pd.DataFrame, allowed_features: list) -> pd.DataFrame:
    sequence_features = ['f__previous_question', 'f__previous_answer',
                         'f__previous_roster', 'f__sequence_jump']
    
    # Filter to only allowed features
    selected_features = [f for f in sequence_features if f in allowed_features]
    
    if selected_features:
        # Select columns to merge
        cols_to_use = ['index_col'] + selected_features
        # Ensure columns exist in df_sequence
        cols_to_use = [c for c in cols_to_use if c in df_sequence.columns]
        
        if len(cols_to_use) > 1: # at least index_col + 1 feature
            df_item = df_item.merge(df_sequence[cols_to_use], how='left', on='index_col')
            
    return df_item

def add_item_time_features(df_item: pd.DataFrame, df_time: pd.DataFrame, allowed_features: list, item_level_columns: list) -> pd.DataFrame:
    time_features = ['f__answer_duration', 'f__comment_duration']
    
    selected_features = [f for f in time_features if f in allowed_features]
    
    if selected_features:
        # Filter out empty variable_name (Pauses)
        df_time_filtered = df_time[df_time['variable_name'] != ''].copy()
        
        # Summarize on item level
        # Note: df_time might have multiple events per item (e.g. AnswerRemoved then AnswerSet)
        # We sum the durations.
        agg_dict = {}
        if 'f__answer_duration' in selected_features:
            agg_dict['f__answer_duration'] = 'sum'
        if 'f__comment_duration' in selected_features:
            agg_dict['f__comment_duration'] = 'sum'
            
        if agg_dict:
             # Ensure grouping columns exist
            group_cols = [c for c in item_level_columns + ['index_col'] if c in df_time_filtered.columns]
            
            df_agg = df_time_filtered.groupby(group_cols).agg(agg_dict).reset_index()
            
            # Merge
            df_agg = df_agg[['index_col'] + list(agg_dict.keys())]
            df_item = df_item.merge(df_agg, how='left', on='index_col')
            
    return df_item

def add_pause_features(df_unit: pd.DataFrame, df_time: pd.DataFrame, allowed_features: list) -> pd.DataFrame:
    pause_features = ['f__pause_count', 'f__pause_duration', 'f__pause_list']
    selected_features = [f for f in pause_features if f in allowed_features]

    if selected_features:
        # Calculate pause stats per interview
        # f__pause_duration column in df_time contains the duration for Resumed/Restarted events
        
        # Custom aggregation for list
        def to_list(x):
            return x.tolist()

        agg_dict = {}
        if 'f__pause_count' in selected_features:
             # count all occurrences (size) where pause_duration is not null is implied by how df_time was built?
             # Actually df_time['f__pause_duration'] is NaN for non-pause events.
             # So we should count non-nulls. 'count' counts non-NA. 'size' counts matches.
             agg_dict['f__pause_count'] = ('f__pause_duration', 'count')
        if 'f__pause_duration' in selected_features:
             agg_dict['f__pause_duration'] = ('f__pause_duration', 'sum')
        if 'f__pause_list' in selected_features:
             # This might be tricky in aggregation if all are NaN. 
             # We filter first.
             pass

        if agg_dict:
            df_pause = df_time.groupby('interview__id').agg(**agg_dict).reset_index()
            
            # Handle list separately if needed or include in agg above if simple
            if 'f__pause_list' in selected_features:
                 # Only rows with valid pause duration
                 pause_rows = df_time.dropna(subset=['f__pause_duration'])
                 if not pause_rows.empty:
                    list_agg = pause_rows.groupby('interview__id')['f__pause_duration'].apply(list).reset_index(name='f__pause_list')
                    df_pause = df_pause.merge(list_agg, how='left', on='interview__id')
            
            df_unit = df_unit.merge(df_pause, how='left', on='interview__id')

    return df_unit

def add_unit_time_features(df_unit: pd.DataFrame, df_time: pd.DataFrame, allowed_features: list) -> pd.DataFrame:
    time_features = ['f__total_duration', 'f__total_elapse', 'f__days_from_start', 'f__time_changed']
    selected_features = [f for f in time_features if f in allowed_features]
    
    if selected_features:
        agg_dict = {}
        if 'f__total_duration' in selected_features:
            agg_dict['f__total_duration'] = ('f__total_duration', 'sum')
        if 'f__total_elapse' in selected_features:
            # Lambda in agg is slower, but compatible.
             agg_dict['f__total_elapse'] = ('timestamp_local', lambda x: (x.max() - x.min()).total_seconds() if not x.empty else 0)
        if 'f__time_changed' in selected_features:
            agg_dict['f__time_changed'] = ('f__time_changed', 'sum')
        if 'f__days_from_start' in selected_features:
            agg_dict['f__days_from_start'] = ('f__days_from_start', 'min')

        if agg_dict:
            df_dur = df_time.groupby('interview__id').agg(**agg_dict).reset_index()
            df_unit = df_unit.merge(df_dur, how='left', on='interview__id')

    return df_unit


# --- Base Table Creation ---

def create_base_item_table(microdata: pd.DataFrame, paradata_active: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Creates the base item table by merging microdata with paradata information.
    Equivalent to FeatureProcessing.make_df_item.
    """
    logger.info("Creating base item table...")
    
    item_level_columns = ['interview__id', 'variable_name', 'roster_level']
    allowed_features = ['f__' + k for k, v in parameters['features'].items() if v.get('use', False)]

    # 1. Create Index Column on Microdata
    df_item = make_index_col(microdata.copy())
    
    # 2. Select initial columns
    initial_cols = ['value', "qtype", 'is_integer', 'qnr_seq',
                    'n_answers', 'answer_sequence',
                    'cascade_from_question_id', 'is_filtered_combobox',
                    'index_col'] + item_level_columns
    
    # Intersect with available columns to avoid KeyErrors
    cols_to_keep = [c for c in initial_cols if c in df_item.columns]
    df_item = df_item[cols_to_keep]

    # 3. Prepare Paradata for Merge
    # We want the *last* AnswerSet for each item
    paradata_columns = ['responsible', 'f__answer_hour_set', 'interviewing', 'tz_offset']
    available_para_cols = [c for c in paradata_columns if c in paradata_active.columns]
    
    answer_set_mask = (paradata_active['event'] == 'AnswerSet')
    
 # # Already present in paradata_active from ingestion, but ensure it's there for merging
    # if 'index_col' not in paradata_active.columns:
    #     paradata_active = make_index_col(paradata_active.copy())
        
    data_to_merge = paradata_active[answer_set_mask].drop_duplicates(subset='index_col', keep='last')
    
    # 4. Merge
    df_item = df_item.merge(data_to_merge[available_para_cols + ['index_col']], how='left', on='index_col')

    # 5. Filter for 'interviewing' == True (Supervisor Logic)
    # Remove items that are not in interviewing
    df_item = df_item[df_item['interviewing'] == True].copy()

    # 6. Add Sequence Features
    # Pre-calculate sequence df
    df_sequence = get_df_sequence(paradata_active)
    df_item = add_sequence_features(df_item, df_sequence, allowed_features)

    # 7. Add Time Features
    # Pre-calculate time df
    df_time = get_df_time(paradata_active)
    df_item = add_item_time_features(df_item, df_time, allowed_features, item_level_columns)

    return df_item

def create_base_unit_table(paradata_active: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Creates the base unit table (one row per interview).
    Equivalent to FeatureProcessing.make_df_unit.
    """
    logger.info("Creating base unit table...")
    allowed_features = ['f__' + k for k, v in parameters['features'].items() if v.get('use', False)]
    
    # 1. Initialize from paradata
    cols = ['interview__id', 'responsible', 'survey_name', 'survey_version']
    cols = [c for c in cols if c in paradata_active.columns]
    
    df_unit = paradata_active[cols].copy()
    df_unit.drop_duplicates(inplace=True)
    
    # Filter valid responsible
    df_unit = df_unit[(df_unit['responsible'] != '') & (df_unit['responsible'].notna())]
    
    # 2. Add Pause Features
    df_time = get_df_time(paradata_active)
    df_unit = add_pause_features(df_unit, df_time, allowed_features)
    
    # 3. Add Unit Time Features
    df_unit = add_unit_time_features(df_unit, df_time, allowed_features)
    
    return df_unit


# --- Feature Enrichment Functions (Item) ---

def feat_string_length(df_item, **kwargs):
    feature_name = 'f__string_length'
    mask = df_item["qtype"] == 'TextQuestion'
    df_item[feature_name] = pd.NA
    if mask.any():
        # Use .str.len() directly to preserve NA (astype(str) would convert NaN -> "nan", length 3)
        df_item.loc[mask, feature_name] = df_item.loc[mask, 'value'].str.len().astype('Int64')
    return df_item

def feat_numeric_response(df_item, **kwargs):
    feature_name = 'f__numeric_response'
    numeric_mask = get_numeric_mask(df_item)
    df_item[feature_name] = np.nan
    if numeric_mask.any():
        df_item.loc[numeric_mask, feature_name] = pd.to_numeric(df_item.loc[numeric_mask, 'value'], errors='coerce')
    return df_item

def feat_first_digit(df_item, **kwargs):
    feature_name = 'f__first_digit'
    numeric_mask = get_numeric_mask(df_item)
    df_item[feature_name] = pd.NA
    if numeric_mask.any():
        # Take absolute value, convert to string, extract first character
        vals = pd.to_numeric(df_item.loc[numeric_mask, 'value']).abs().astype(str).str[0]
        df_item.loc[numeric_mask, feature_name] = pd.to_numeric(vals, errors='coerce').astype('Int64')
    return df_item

def feat_last_digit(df_item, **kwargs):
    feature_name = 'f__last_digit'
    # Use the same mask as legacy: excludes empty, null, and -999999999
    numeric_mask = get_numeric_mask(df_item)
    df_item[feature_name] = pd.NA

    if numeric_mask.any():
        # Cast to Int64 (nullable) matching legacy astype('int64'), then apply x >= 1 check
        # Note: legacy checks x >= 1 (not abs(x) >= 1), so negative values correctly yield NA
        vals = pd.to_numeric(df_item.loc[numeric_mask, 'value']).astype('Int64')
        # .where(condition) keeps values where True, sets False to NA
        df_item.loc[numeric_mask, feature_name] = (vals % 10).where(vals >= 1)

    return df_item

def feat_first_decimal(df_item, **kwargs):
    feature_name = 'f__first_decimal'
    # mask: not integer and not empty
    mask = (df_item['is_integer'] == False) & (df_item['value'] != '')
    df_item[feature_name] = pd.NA
    
    if mask.any():
        values = pd.to_numeric(df_item.loc[mask, 'value'], errors='coerce')
        # floor(val * 100) % 100 ?? Legacy code: np.floor(values * 100) % 100
        # This actually gets the first two decimals?
        # Example: 0.123 -> 12.3 -> 12.
        # Wait, if I want first decimal digit (e.g. 1 in 0.123): floor(val * 10) % 10
        # Documentation says "first decimal digit". Code says *100 % 100.
        # I will strictly follow legacy code logic.
        res = np.floor(values * 100) % 100
        df_item.loc[mask, feature_name] = res.astype('Int64')
        
    return df_item

def feat_answer_position(df_item, **kwargs):
    feature_name = 'f__answer_position' # in legacy it was f__rel_answer_position sometimes? code says f__answer_position
    
    # filters
    mask = ((df_item["qtype"] == 'SingleQuestion')
            & (df_item['n_answers'] > 2)
            & (df_item['is_filtered_combobox'] == False)
            & (df_item['cascade_from_question_id'].isna()))
            
    df_item[feature_name] = np.nan
    
    if mask.any():
        # logic: index of value in answer_sequence / (n_answers - 1)
        # answer_sequence is typically a list or string representation of list
        # We need to iterate or apply
        
        def calc_pos(row):
            val = row['value']
            seq = row['answer_sequence']
            n = row['n_answers']
            if isinstance(seq, list) and val in seq:
                 try:
                    idx = seq.index(val)
                    return round(idx / (n - 1), 3)
                 except:
                    return None
            return None

        # Apply is slow but robust for list operations in cells
        df_item.loc[mask, feature_name] = df_item.loc[mask].apply(calc_pos, axis=1)
        
    return df_item

def feat_answer_changed(df_item, **kwargs):
    """
    ⚠️ Legacy bug fixed: the legacy code applied the yes_list change
    check and immediately overwrote it with the no_list check (two separate .loc assignments
    on the same mask), so yes_list changes were always ignored. This implementation
    combines both checks using a bitwise OR.
    """
    feature_name = 'f__answer_changed'
    paradata_active = kwargs.get('paradata_active')

    if paradata_active is None:
        return df_item

    item_level_columns = ['interview__id', 'variable_name', 'roster_level']
    df_changed = paradata_active[paradata_active['event'] == 'AnswerSet'].copy()

    if 'index_col' not in df_changed.columns:
        df_changed = make_index_col(df_changed)

    df_changed[feature_name] = False
    group_cols = [c for c in item_level_columns + ['index_col'] if c in df_changed.columns]
    has_yes_no = 'yes_no_view' in df_changed.columns

    # --- Case 1: TextListQuestion and MultyOptionsQuestion (without yes_no_view mode) ---
    list_mask = (df_changed["qtype"] == 'TextListQuestion')
    multi_mask = (df_changed['yes_no_view'] == False) if has_yes_no else pd.Series(False, index=df_changed.index)

    df_changed['answer_list'] = pd.NA
    df_changed.loc[list_mask, 'answer_list'] = df_changed.loc[list_mask, 'answer'].str.split('|')
    df_changed.loc[multi_mask, 'answer_list'] = df_changed.loc[multi_mask, 'answer'].str.split(r', |\|')

    df_changed['prev_answer_list'] = df_changed.groupby(group_cols)['answer_list'].shift()
    answers_mask = df_changed['prev_answer_list'].notna()
    if answers_mask.any():
        df_changed.loc[answers_mask, feature_name] = df_changed.loc[answers_mask].apply(
            lambda row: not set(row['prev_answer_list']).issubset(set(row['answer_list'])), axis=1
        )

    # --- Case 2: Single-answer questions ---
    df_changed['prev_answer'] = df_changed.groupby(group_cols)['answer'].shift()
    single_answer_mask = (
        (~df_changed["qtype"].isin(['MultyOptionsQuestion', 'TextListQuestion'])) &
        (df_changed['prev_answer'].notna()) &
        (df_changed['answer'] != df_changed['prev_answer'])
    )
    df_changed.loc[single_answer_mask, feature_name] = True

    # --- Case 3: Yes/No view questions ---
    if has_yes_no:
        yesno_mask = (df_changed['yes_no_view'] == True)
        if yesno_mask.any():
            df_filtered = df_changed[yesno_mask].copy()
            df_filtered[['yes_list', 'no_list']] = df_filtered['answer'].str.split('|', expand=True)
            df_filtered['yes_list'] = df_filtered['yes_list'].str.split(', ').apply(
                lambda x: [] if x == [''] or x is None else x)
            df_filtered['no_list'] = df_filtered['no_list'].str.split(', ').apply(
                lambda x: [] if x == [''] or x is None else x)
            yesno_group_cols = [c for c in group_cols if c in df_filtered.columns]
            df_filtered['prev_yes_list'] = df_filtered.groupby(yesno_group_cols)['yes_list'].shift(fill_value=[])
            df_filtered['prev_no_list'] = df_filtered.groupby(yesno_group_cols)['no_list'].shift(fill_value=[])
            # A change occurs if either yes or no selections changed
            yes_changed = df_filtered.apply(
                lambda row: not set(row['prev_yes_list']).issubset(set(row['yes_list'])), axis=1)
            no_changed = df_filtered.apply(
                lambda row: not set(row['prev_no_list']).issubset(set(row['no_list'])), axis=1)
            df_changed.loc[yesno_mask, feature_name] = (yes_changed | no_changed).values

    # Sum changes per item and map back
    changes_per_item = df_changed.groupby('index_col')[feature_name].sum()
    df_item[feature_name] = df_item['index_col'].map(changes_per_item).fillna(0)

    return df_item

def feat_answer_selected(df_item, **kwargs):
    feature_name = 'f__answer_selected'
    mask = df_item["qtype"].isin(['MultyOptionsQuestion'])
    
    df_item[feature_name] = np.nan
    
    # Value is list? Or string? Usually lists in newer pandas if parquet preserved it, 
    # but legacy often had strings.
    # Assuming value is list if parquet
    
    if mask.any():
        def count_els(x):
            if isinstance(x, list): return len(x)
            if isinstance(x, str): return len(x.split('|')) # simple heuristic for pipe-sep
            return np.nan
            
        df_item.loc[mask, feature_name] = df_item.loc[mask, 'value'].apply(count_els)
        # Ratio
        df_item.loc[mask, feature_name] = df_item.loc[mask, feature_name] / df_item.loc[mask, 'n_answers']
        
    return df_item

def feat_gps(df_item, **kwargs):
    # Sets f__gps boolean flag plus f__gps_latitude, f__gps_longitude, f__gps_accuracy
    feature_name = 'f__gps'
    mask = df_item["qtype"] == 'GpsCoordinateQuestion'
    df_item[feature_name] = False
    if mask.any():
        df_item.loc[mask, feature_name] = True
        # Split value "lat,lon,acc,alt,timestamp_utc"
        gps_data = df_item.loc[mask, 'value'].str.split(',', expand=True)
        if gps_data.shape[1] >= 3:
            df_item.loc[mask, 'f__gps_latitude'] = pd.to_numeric(gps_data[0], errors='coerce')
            df_item.loc[mask, 'f__gps_longitude'] = pd.to_numeric(gps_data[1], errors='coerce')
            df_item.loc[mask, 'f__gps_accuracy'] = pd.to_numeric(gps_data[2], errors='coerce')
    return df_item


def feat_comment_length(df_item, **kwargs):
    """Total character length of all comments left on each item.
    Matches legacy make_feature_item__comment_length which uses self.df_paradata
    (all events, role=1, interviewing=True — not limited to active events).
    """
    feature_name = 'f__comment_length'
    paradata_full = kwargs.get('paradata_full')
    if paradata_full is None:
        return df_item

    comment_mask = (
        (paradata_full['event'] == 'CommentSet') &
        (paradata_full['role'] == 1)
    )
    df_comment = paradata_full[comment_mask].copy()
    if df_comment.empty:
        return df_item

    if 'index_col' not in df_comment.columns:
        df_comment = make_index_col(df_comment)

    df_comment[feature_name] = df_comment['answer'].str.len()
    df_agg = df_comment.groupby('index_col').agg(f__comment_length=(feature_name, 'sum'))
    df_item[feature_name] = df_item['index_col'].map(df_agg['f__comment_length'])
    return df_item


def feat_comment_set(df_item, **kwargs):
    """Count of CommentSet events per item.
    Matches legacy make_feature_item__comment_set which uses self.df_paradata
    (all events, role=1, interviewing=True — not limited to active events).
    """
    feature_name = 'f__comment_set'
    paradata_full = kwargs.get('paradata_full')
    if paradata_full is None:
        return df_item

    comment_mask = (
        (paradata_full['event'] == 'CommentSet') &
        (paradata_full['role'] == 1)
    )
    df_comment = paradata_full[comment_mask].copy()
    if df_comment.empty:
        return df_item

    if 'index_col' not in df_comment.columns:
        df_comment = make_index_col(df_comment)

    df_agg = df_comment.groupby('index_col').agg(f__comment_set=('order', 'count'))
    df_item[feature_name] = df_item['index_col'].map(df_agg['f__comment_set'])
    return df_item


def feat_answer_removed(df_item, **kwargs):
    """Count of AnswerRemoved events per item.
    Matches legacy get_feature_item__answer_removed which uses self.df_paradata
    (all events, role=1, interviewing=True — not limited to active events).
    The legacy method notes this feature may include items no longer in microdata.
    """
    feature_name = 'f__answer_removed'
    paradata_full = kwargs.get('paradata_full')
    if paradata_full is None:
        return df_item

    removed_mask = (
        (paradata_full['event'] == 'AnswerRemoved') &
        (paradata_full['role'] == 1)
    )
    df_removed = paradata_full[removed_mask]
    if df_removed.empty:
        return df_item

    # Legacy groups on interview__id + responsible + variable_name + qnr_seq;
    # we merge on interview__id + variable_name which is safe since qnr_seq is 1:1 with variable_name.
    df_agg = df_removed.groupby(
        ['interview__id', 'variable_name']
    ).agg(f__answer_removed=('order', 'count')).reset_index()

    df_item = df_item.merge(
        df_agg[['interview__id', 'variable_name', feature_name]],
        how='left',
        on=['interview__id', 'variable_name']
    )
    return df_item


# Dispatcher
ITEM_FEATURE_MAP = {
    'string_length': feat_string_length,
    'numeric_response': feat_numeric_response,
    'first_digit': feat_first_digit,
    'last_digit': feat_last_digit,
    'first_decimal': feat_first_decimal,
    'answer_position': feat_answer_position,
    'answer_changed': feat_answer_changed,
    'answer_selected': feat_answer_selected,
    'answer_removed': feat_answer_removed,
    'comment_length': feat_comment_length,
    'comment_set': feat_comment_set,
    'gps': feat_gps,
}


def enrich_item_features(df_item: pd.DataFrame, paradata_active: pd.DataFrame, paradata_full: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Applies feature engineering logic to the item table.
    paradata_active: active interviewer events (self.df_active_paradata equivalent).
    paradata_full: all processed events, role=1, interviewing=True (self.df_paradata equivalent).
    """
    logger.info("Enriching item features...")
    allowed_features = parameters.get('features', {})

    # Ensure index_col in paradata for lookups
    if 'index_col' not in paradata_active.columns:
        paradata_active = make_index_col(paradata_active.copy())
    if 'index_col' not in paradata_full.columns:
        paradata_full = make_index_col(paradata_full.copy())

    for feat_key, feat_cfg in allowed_features.items():
        if feat_cfg.get('use', False):
            func = ITEM_FEATURE_MAP.get(feat_key)
            if func:
                logger.info(f"Calculating item feature: {feat_key}")
                try:
                    df_item = func(df_item, paradata_active=paradata_active, paradata_full=paradata_full)
                except Exception as e:
                    logger.warning(f"Failed to calculate {feat_key}: {e}")

    return df_item


# --- Feature Enrichment Functions (Unit) ---

def feat_unit_number_answered(df_unit, item_features, **kwargs):
    feature_name = 'f__number_answered'
    # Match legacy make_feature_unit__number_answered: exclude null, -999999999, '##N/A##',
    # empty string, and Variable-type questions
    mask = (
        (~pd.isnull(item_features['value'])) &
        (item_features['value'] != -999999999) &
        (item_features['value'] != '##N/A##') &
        (item_features['value'] != '') &
        (item_features['qtype'] != 'Variable')
    )
    df_agg = item_features[mask].groupby('interview__id').agg(
        f__number_answered=('value', 'count')
    )
    df_unit[feature_name] = df_unit['interview__id'].map(df_agg['f__number_answered']).fillna(0)
    return df_unit

def feat_unit_number_unanswered(df_unit, item_features, **kwargs):
    feature_name = 'f__number_unanswered'
    # Match legacy make_feature_unit__number_unanswered: -999999999 or '##N/A##', excluding Variable type
    mask = (
        (
            (item_features['value'] == -999999999) |
            (item_features['value'] == '##N/A##')
        ) &
        (item_features['qtype'] != 'Variable')
    )
    df_agg = item_features[mask].groupby('interview__id').agg(
        f__number_unanswered=('value', 'count')
    )
    df_unit[feature_name] = df_unit['interview__id'].map(df_agg['f__number_unanswered']).fillna(0)
    return df_unit

def feat_unit_translation_positions(df_unit, item_features, **kwargs):
    """Relative positions of TranslationSwitched events within each interview.
    Matches legacy make_feature_unit__translation_positions which uses self.df_paradata.

    Returns a list of relative positions (0..1) per interview.
    """
    feature_name = 'f__translation_positions'
    paradata_full = kwargs.get('paradata_full')
    if paradata_full is None:
        return df_unit

    trans_mask = paradata_full['event'].isin(['AnswerSet', 'TranslationSwitched'])
    df_trans = paradata_full.loc[trans_mask, ['interview__id', 'order', 'event', 'param']].copy()
    if df_trans.empty:
        return df_unit

    df_trans = df_trans.sort_values(['interview__id', 'order']).reset_index(drop=True)
    df_trans['seq'] = df_trans.groupby('interview__id').cumcount() + 1

    def relative_translation_positions(group):
        total_rows = len(group)
        translation_positions = group.loc[group['event'] == 'TranslationSwitched', 'seq']
        return [pos / total_rows for pos in translation_positions]

    # include_groups=False: 'interview__id' is not needed inside the function; suppresses FutureWarning
    # in pandas 2.2+ where including the grouping column in the passed group is deprecated.
    result = df_trans.groupby('interview__id').apply(
        relative_translation_positions, include_groups=False
    ).reset_index()
    result.columns = ['interview__id', feature_name]

    df_unit[feature_name] = df_unit['interview__id'].map(
        result.set_index('interview__id')[feature_name]
    )
    return df_unit


UNIT_FEATURE_MAP = {
    'number_answered': feat_unit_number_answered,
    'number_unanswered': feat_unit_number_unanswered,
    'translation_positions': feat_unit_translation_positions,
}

def enrich_unit_features(df_unit: pd.DataFrame, item_features: pd.DataFrame, paradata_full: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Applies feature engineering logic to the unit table.
    paradata_full: all processed events, role=1, interviewing=True (self.df_paradata equivalent).
    Required for f__translation_positions.
    """
    logger.info("Enriching unit features...")
    allowed_features = parameters.get('features', {})

    for feat_key, feat_cfg in allowed_features.items():
        if feat_cfg.get('use', False):
            func = UNIT_FEATURE_MAP.get(feat_key)
            if func:
                logger.info(f"Calculating unit feature: {feat_key}")
                try:
                    df_unit = func(df_unit, item_features=item_features, paradata_full=paradata_full)
                except Exception as e:
                    logger.warning(f"Failed to calculate {feat_key}: {e}")

    return df_unit

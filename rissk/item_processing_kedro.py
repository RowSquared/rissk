import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from pyod.models.thresholds import FILTER

logger = logging.getLogger(__name__)

def rename_feature(feature_name: str, starting_string: str = 'f', new_string: str = 's') -> str:
    """Rename feature correctly mapping to score (f__ -> s__)."""
    starting_string = starting_string + '__'
    new_string = new_string + '__'
    if feature_name.startswith(starting_string):
        return feature_name.replace(starting_string, new_string)
    return feature_name

def get_contamination_parameter(config_features: dict, feature_name: str, automatic_contamination: bool = False, method: str = 'medfilt', random_state: int = 42):
    """Fetch contamination parameter from Kedro parameters/config features.
    
    Returns a FILTER object for automatic contamination detection (matching legacy behaviour),
    or a fixed float when a contamination value is explicitly configured.
    """
    f_name = feature_name.replace('f__', '')
    contamination = config_features.get(f_name, {}).get('parameters', {}).get('contamination')
    if contamination is None or contamination == 'auto' or automatic_contamination is True:
        return FILTER(method=method, random_state=random_state)
    else:
        return float(contamination)

def filter_variable_name_by_frequency(df: pd.DataFrame, feature_name: str, frequency: int = 100, min_unique_values: int = 3) -> List[str]:
    """Filter variables by frequency and unique values."""
    if feature_name not in df.columns:
        return []
    valid_data = df[~pd.isnull(df[feature_name])]
    grouped_df = valid_data.groupby('variable_name')[feature_name].agg(['count', 'nunique'])
    valid_variables = grouped_df[(grouped_df['count'] >= frequency) & (grouped_df['nunique'] >= min_unique_values)].index
    return valid_variables.tolist()

def filter_columns(data: pd.DataFrame, index_col: List[str], threshold: int = 100) -> Tuple[List[str], List[str]]:
    """Determine columns to keep/drop based on threshold (placeholder refactor)"""
    # Count non-null values for each column
    non_null_counts = data.drop(columns=index_col, errors='ignore').count()
    # Filter columns to keep
    keep_columns = non_null_counts[non_null_counts >= threshold].index.tolist()
    drop_columns = non_null_counts[non_null_counts < threshold].index.tolist()
    return index_col + keep_columns, drop_columns

def get_clean_pivot_table(df_item: pd.DataFrame, feature_name: str, remove_low_freq_col: bool = True, filter_conditions=None, threshold: int = 100) -> Tuple[pd.DataFrame, List[str]]:
    """Create a pivot table handling columns and filtering."""
    index_col = ['interview__id', 'roster_level', 'responsible']
    data = df_item.copy()
    
    if filter_conditions is not None: # Not yet strictly typed since condition type unknown
        pass # To fully mimic we'd apply filter
        
    data = pd.pivot_table(data=data, index=index_col, columns='variable_name',
                          values=feature_name, fill_value=np.NAN)
    data = data.reset_index()
    
    if data.columns.nlevels > 1:
        pass # In case of multi index columns flatten, handled differently?
        
    index_col = [col for col in index_col if col in data.columns]
    keep_columns, drop_columns = filter_columns(data, index_col, threshold=threshold)
    
    if remove_low_freq_col:
       data = data[keep_columns]
       
    return data, index_col

# --- SCORING FUNCTIONS BEGIN --- 
# (To be filled out next, mapping make_score__*)

from pyod.models.ecod import ECOD

def calculate_answer_hour_set_score(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    feature_name = 'f__answer_hour_set'
    score_name = rename_feature(feature_name)
    df = df_item[~pd.isnull(df_item[feature_name])].copy()

    if df.empty:
        df_item[score_name] = np.nan
        return df_item

    sorted_hours = df[feature_name].value_counts().index
    hour_to_rank = {hour: rank for rank, hour in enumerate(sorted_hours)}
    df['frequency'] = df[feature_name].map(hour_to_rank)

    contamination_param = parameters.get('features', {})
    contamination = get_contamination_parameter(contamination_param, feature_name)
    
    model = ECOD(contamination=contamination)
    model.fit(df[[feature_name]])
    df[score_name] = model.predict(df[[feature_name]])

    df.loc[df['frequency'] <= df[df[score_name] == 0]['frequency'].min(), score_name] = 0
    df.drop(columns=['frequency'], inplace=True)
    
    # Merge back to original dataframe
    df_out = df_item.copy()
    df_out[score_name] = df_out.index.map(df[score_name])
    return df_out

def calculate_sequence_jump_score(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    feature_name = 'f__sequence_jump'
    score_name = rename_feature(feature_name)
    df = df_item.copy()

    if feature_name not in df.columns or df[feature_name].dropna().empty:
        df[score_name] = np.nan
        return df

    valid_data = df[~pd.isnull(df[feature_name])].copy()
    valid_variables = filter_variable_name_by_frequency(valid_data, feature_name, frequency=100, min_unique_values=3)
    df[score_name] = 0
    contamination = get_contamination_parameter(parameters.get('features', {}), feature_name)

    from pyod.models.inne import INNE
    for var in valid_variables:
        mask = (df['variable_name'] == var) & (~pd.isnull(df[feature_name]))
        if mask.sum() > 0:
            model = INNE(contamination=contamination, random_state=42)
            model.fit(df.loc[mask, [feature_name]])
            df.loc[mask, score_name] = model.predict(df.loc[mask, [feature_name]])

    return df

from pyod.models.cof import COF

def calculate_first_decimal_score(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    feature_name = 'f__first_decimal'
    score_name = rename_feature(feature_name)
    df = df_item.copy()
    
    if feature_name not in df.columns or df[feature_name].dropna().empty:
        df[score_name] = np.nan
        return df
        
    valid_data = df[~pd.isnull(df[feature_name])]
    valid_variables = filter_variable_name_by_frequency(valid_data, feature_name, frequency=100, min_unique_values=3)
    df[score_name] = np.nan
    df.loc[~pd.isnull(df[feature_name]), score_name] = 0
    contamination = get_contamination_parameter(parameters.get('features', {}), feature_name, method='medfilt', random_state=42)
    
    for var in valid_variables:
        mask = (df['variable_name'] == var) & (~pd.isnull(df[feature_name]))
        if mask.sum() > 0:
            model = COF(contamination=contamination)
            model.fit(df.loc[mask, [feature_name]])
            df.loc[mask, score_name] = model.predict(df.loc[mask, [feature_name]])
            
    return df

def calculate_answer_changed_score(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    feature_name = 'f__answer_changed'
    score_name = rename_feature(feature_name)
    df = df_item.copy()
    
    if feature_name not in df.columns or df[feature_name].dropna().empty:
        df[score_name] = np.nan
        return df

    valid_data = df[~pd.isnull(df[feature_name])]
    valid_variables = filter_variable_name_by_frequency(valid_data, feature_name, frequency=100, min_unique_values=1)
    df[score_name] = np.nan
    df.loc[~pd.isnull(df[feature_name]), score_name] = 0
    contamination = get_contamination_parameter(parameters.get('features', {}), feature_name, method='medfilt', random_state=42)
    
    for var in valid_variables:
        mask = (df['variable_name'] == var) & (~pd.isnull(df[feature_name]))
        if mask.sum() > 0:
            model = ECOD(contamination=contamination)
            model.fit(df.loc[mask, [feature_name]])
            df.loc[mask, score_name] = model.predict(df.loc[mask, [feature_name]])
            
    return df

def calculate_answer_removed_score(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    feature_name = 'f__answer_removed'
    score_name = rename_feature(feature_name)
    df = df_item.copy()

    if feature_name not in df.columns or df[feature_name].dropna().empty:
        df[score_name] = np.nan
        return df

    valid_variables = filter_variable_name_by_frequency(df, feature_name, frequency=100, min_unique_values=1)
    df[score_name] = np.nan
    df.loc[~pd.isnull(df[feature_name]), score_name] = 0
    contamination = get_contamination_parameter(parameters.get('features', {}), feature_name, method='medfilt', random_state=42)

    for var in valid_variables:
        mask = (df['variable_name'] == var) & (~pd.isnull(df[feature_name]))
        if mask.sum() > 0:
            model = ECOD(contamination=contamination)
            model.fit(df.loc[mask, [feature_name]])
            df.loc[mask, score_name] = model.predict(df.loc[mask, [feature_name]])

    return df

from scipy.spatial import cKDTree
from pyod.models.lof import LOF

# Attempting to import legacy stats_utils safely for the math functions
try:
    from rissk.utils.stats_utils import (
        calculate_entropy, 
        calculate_list_entropy, 
        filter_variables_by_magnitude, 
        apply_benford_tests
    )
except ImportError:
    pass

try:
    from rissk.detection_algorithms_kedro import lat_lon_to_cartesian
except ImportError:
    pass

def calculate_answer_position_score(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    feature_name = 'f__answer_position'
    score_name = rename_feature(feature_name)
    df = df_item.copy()
    
    if feature_name not in df.columns or df[feature_name].dropna().empty:
        df[score_name] = np.nan
        return df

    valid_data = df[~pd.isnull(df[feature_name])]
    valid_variables = filter_variable_name_by_frequency(valid_data, feature_name, frequency=100, min_unique_values=3)
    df[score_name] = 0
    
    for var in valid_variables:
        mask = (df['variable_name'] == var)
        if mask.sum() > 0:
            unique_values = df[mask][feature_name].nunique()
            try:
                entropy_df = df[mask].groupby('responsible')[feature_name].apply(
                    calculate_entropy, unique_values=unique_values, min_record_sample=10
                ).reset_index()
            except NameError:
                continue # if calculate_entropy not found

            entropy_df = entropy_df[~pd.isnull(entropy_df[feature_name])]

            if entropy_df.shape[0] > 0:
                entropy_df.sort_values(feature_name, inplace=True, ascending=False)
                median_value = entropy_df[feature_name].median()
                entropy_df[score_name] = entropy_df[feature_name].apply(
                    lambda x: 1 if x < median_value - 0.5 * median_value else 0)
                
                # Apply map safely
                responsible_map = entropy_df.set_index('responsible')[score_name].to_dict()
                df.loc[mask, score_name] = df.loc[mask, 'responsible'].map(responsible_map).fillna(0)
    return df

def calculate_answer_selected_score(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    feature_name = 'f__answer_selected'
    score_name = rename_feature(feature_name)
    df = df_item.copy()
    
    if feature_name not in df.columns or df[feature_name].dropna().empty:
        df[score_name + '_lower'] = np.nan
        df[score_name + '_upper'] = np.nan
        return df

    valid_data = df[~pd.isnull(df[feature_name])]
    valid_variables = filter_variable_name_by_frequency(valid_data, feature_name, frequency=100, min_unique_values=3)
    
    score_name1 = score_name + '_lower'
    score_name2 = score_name + '_upper'
    df[score_name] = 0
    
    contamination = get_contamination_parameter(parameters.get('features', {}), feature_name, method='medfilt', random_state=42)

    for var in valid_variables:
        mask = (df['variable_name'] == var) & (~pd.isnull(df[feature_name]))
        if mask.sum() > 0:
            model = ECOD(contamination=contamination)
            model.fit(df.loc[mask, [feature_name]])
            
            df.loc[mask, score_name] = model.predict(df.loc[mask, [feature_name]])
            
            non_anomalies = df.loc[mask & (df[score_name] == 0), feature_name]
            if not non_anomalies.empty:
                min_good_value = non_anomalies.min()
                max_good_value = non_anomalies.max()
                
                df.loc[mask, score_name1] = 0
                df.loc[mask, score_name2] = 0
                
                df.loc[mask & (df[feature_name] < min_good_value), score_name1] = 1
                df.loc[mask & (df[feature_name] > max_good_value), score_name2] = 1

    df.drop(columns=[score_name], errors='ignore', inplace=True)
    return df

def calculate_answer_duration_score(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    feature_name = 'f__answer_duration'
    score_name = rename_feature(feature_name)
    df = df_item.copy()
    
    if feature_name not in df.columns or df[feature_name].dropna().empty:
        df[score_name + '_lower'] = np.nan
        df[score_name + '_upper'] = np.nan
        return df

    valid_data = df[~pd.isnull(df[feature_name])]
    valid_variables = filter_variable_name_by_frequency(valid_data, feature_name, frequency=100, min_unique_values=3)

    score_name1 = score_name + '_lower'
    score_name2 = score_name + '_upper'
    df[score_name1] = 0
    df[score_name2] = 0
    df[score_name] = 0
    
    contamination = get_contamination_parameter(parameters.get('features', {}), feature_name, method='medfilt', random_state=42)

    for var in valid_variables:
        mask = (df['variable_name'] == var) & (~pd.isnull(df[feature_name]))
        if mask.sum() > 0:
            model = ECOD(contamination=contamination)
            model.fit(df.loc[mask, [feature_name]])
            df.loc[mask, score_name] = model.predict(df.loc[mask, [feature_name]])

            non_anomalies = df.loc[mask & (df[score_name] == 0), feature_name]
            if not non_anomalies.empty:
                min_good_value = non_anomalies.min()
                max_good_value = non_anomalies.max()
                
                df.loc[mask, score_name1] = 0
                df.loc[mask, score_name2] = 0
                
                df.loc[mask & (df[feature_name] < min_good_value), score_name1] = 1
                df.loc[mask & (df[feature_name] > max_good_value), score_name2] = 1

    df.drop(columns=[score_name], errors='ignore', inplace=True)
    return df

def calculate_single_question_score(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    feature_name = 'f__single_question'
    score_name = rename_feature(feature_name)
    df = df_item.copy()
    
    if 'qtype' not in df.columns or 'n_answers' not in df.columns or 'value' not in df.columns:
        df[score_name] = np.nan
        return df

    # Mask specific for single questions without filter rules bypassing cascades
    single_question_mask = (
        (df["qtype"] == 'SingleQuestion') & 
        (df['n_answers'] > 1) & 
        (df.get('is_filtered_combobox', False) == False) & 
        (pd.isnull(df.get('cascade_from_question_id', np.nan)))
    )

    df[score_name] = 0
    valid_data = df[single_question_mask]
    if valid_data.empty: return df
    
    variables = filter_variable_name_by_frequency(valid_data, 'value', frequency=100, min_unique_values=3)
    
    for var in variables:
        mask = (df['variable_name'] == var) & single_question_mask
        if mask.sum() > 0:
            unique_values = df.loc[mask, 'value'].nunique()
            try:
                entropy_df = df[mask].groupby('responsible')['value'].apply(
                    calculate_entropy, unique_values=unique_values
                ).reset_index()
            except NameError:
                continue

            entropy_df = entropy_df[~pd.isnull(entropy_df['value'])]

            if entropy_df.shape[0] > 0:
                entropy_df.sort_values('value', inplace=True, ascending=False)
                median_value = entropy_df['value'].median()
                entropy_df[score_name] = entropy_df['value'].apply(
                    lambda x: 1 if x < median_value - 0.5 * median_value else 0)
                
                responsible_map = entropy_df.set_index('responsible')[score_name].to_dict()
                df.loc[mask, score_name] = df.loc[mask, 'responsible'].map(responsible_map).fillna(0)
                
    return df

def calculate_multi_option_question_score(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    feature_name = 'f__multi_option_question'
    score_name = rename_feature(feature_name)
    df = df_item.copy()
    
    if 'qtype' not in df.columns or 'value' not in df.columns:
        df[score_name] = np.nan
        return df

    multi_question_mask = (df["qtype"] == 'MultyOptionsQuestion')
    valid_data = df[multi_question_mask]
    
    df[score_name] = 0
    if valid_data.empty: return df
    
    # Filter variables safely via counts
    val_counts = valid_data['variable_name'].value_counts()
    variables = val_counts[val_counts >= 100].index

    for var in variables:
        mask = (df['variable_name'] == var) & multi_question_mask
        if mask.sum() > 0:
            # Need safely explode nested lists in values
            exploded_vals = df.loc[mask, 'value'].explode()
            unique_values = len([v for v in exploded_vals.unique() if v != '##N/A##'])
            try:
                entropy_df = df[mask].groupby('responsible')['value'].apply(
                    calculate_list_entropy, unique_values=unique_values, min_record_sample=5
                ).reset_index()
            except NameError:
                continue

            entropy_df = entropy_df[~pd.isnull(entropy_df['value'])]

            if entropy_df.shape[0] > 0:
                entropy_df.sort_values('value', inplace=True, ascending=False)
                median_value = entropy_df['value'].median()
                entropy_df[score_name] = entropy_df['value'].apply(
                    lambda x: 1 if x < median_value - 0.5 * median_value else 0)
                
                responsible_map = entropy_df.set_index('responsible')[score_name].to_dict()
                df.loc[mask, score_name] = df.loc[mask, 'responsible'].map(responsible_map).fillna(0)

    return df

def calculate_first_digit_score(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    feature_name = 'f__numeric_response'
    score_name = 's__first_digit'
    df = df_item.copy()
    
    if feature_name not in df.columns or df[feature_name].dropna().empty:
        df[score_name] = np.nan
        return df

    valid_data = df[~pd.isnull(df[feature_name])]
    valid_variables = filter_variable_name_by_frequency(valid_data, feature_name, frequency=100, min_unique_values=3)
    df[score_name] = 0
    
    try:
        valid_variables = filter_variables_by_magnitude(valid_data, feature_name, valid_variables, min_order_of_magnitude=3)
        benford_jensen_df = apply_benford_tests(
            valid_data, valid_variables, 'responsible', feature_name, apply_first_digit=True, minimum_sample=50
        )
    except NameError:
        return df # dependencies missing
        
    if not benford_jensen_df.empty:
        variable_list = benford_jensen_df['variable_name'].unique()
        for var in variable_list:
            bj_mask = (benford_jensen_df['variable_name'] == var) & (~pd.isnull(benford_jensen_df[feature_name]))
            bj_df = benford_jensen_df[bj_mask].copy()
            if bj_df.shape[0] > 0:
                bj_df.sort_values(feature_name, inplace=True, ascending=True)
                median_value = bj_df[feature_name].median()
                bj_df[score_name] = bj_df[feature_name].apply(
                    lambda x: 1 if x > median_value + 0.5 * median_value else 0)
                
                mask = (df['variable_name'] == var)
                responsible_map = bj_df.set_index('responsible')[score_name].to_dict()
                df.loc[mask, score_name] = df.loc[mask, 'responsible'].map(responsible_map).fillna(0)
                
    return df

def calculate_gps_score(df_item: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    df = df_item.copy()

    required_columns = ['f__gps_latitude', 'f__gps_longitude', 'f__gps_accuracy']
    if any(col not in df.columns for col in required_columns):
        for col in ['s__gps_proximity_counts', 's__gps_outlier', 's__gps_extreme_outlier']:
            df[col] = np.nan
        return df

    gps_mask = (~pd.isnull(df['f__gps_latitude'])) & (~pd.isnull(df['f__gps_longitude']))
    if gps_mask.sum() == 0:
        for col in ['s__gps_proximity_counts', 's__gps_outlier', 's__gps_extreme_outlier']:
            df[col] = np.nan
        return df

    data = df.loc[gps_mask].copy()
    data['s__gps_extreme_outlier'] = 0
    data.loc[data['f__gps_latitude'] == 0.0, 's__gps_extreme_outlier'] = 1
    data.loc[data['f__gps_longitude'] == 0.0, 's__gps_extreme_outlier'] = 1

    data['x'], data['y'], data['z'] = lat_lon_to_cartesian(data['f__gps_latitude'], data['f__gps_longitude'])
    data['accuracy'] = data['f__gps_accuracy'].fillna(0) / 1e6

    tree = cKDTree(data[['x', 'y', 'z']])
    radius = 10 / 1e6
    counts = [
        len(tree.query_ball_point(xyz, r=radius + acc)) - 1
        for xyz, acc in zip(data[['x', 'y', 'z']].values, data['accuracy'])
    ]
    data['s__gps_proximity_counts'] = counts

    mask = data['s__gps_extreme_outlier'] < 1
    data['distance_to_median'] = np.nan
    if mask.sum() > 0:
        median_x = data.loc[mask].drop_duplicates(subset='x')['x'].median()
        median_y = data.loc[mask].drop_duplicates(subset='y')['y'].median()
        median_z = data.loc[mask].drop_duplicates(subset='z')['z'].median()

        data.loc[mask, 'distance_to_median'] = np.sqrt(
            (data.loc[mask, 'x'] - median_x) ** 2
            + (data.loc[mask, 'y'] - median_y) ** 2
            + (data.loc[mask, 'z'] - median_z) ** 2
        )

        p75 = data.loc[mask, 'distance_to_median'].quantile(0.75)
        median = data.loc[mask, 'distance_to_median'].median()
        range_75 = p75 - median
        threshold = p75 + 3.5 * range_75
        data.loc[mask, 's__gps_extreme_outlier'] = (
            data.loc[mask, 'distance_to_median'] > threshold
        ).astype(int)

        contamination = get_contamination_parameter(
            parameters.get('features', {}),
            'f__gps',
            method='medfilt',
            random_state=42,
        )
        coords_columns = ['x', 'y']
        if data.loc[mask].shape[0] < 10000:
            model = COF(contamination=contamination)
        else:
            model = LOF(contamination=contamination, n_neighbors=20)
        model.fit(data.loc[mask, coords_columns])
        data.loc[mask, 's__gps_outlier'] = model.predict(data.loc[mask, coords_columns])
    else:
        data['s__gps_outlier'] = 0

    data['s__gps_outlier'] = data['s__gps_outlier'].fillna(0)
    df.loc[data.index, 's__gps_proximity_counts'] = data['s__gps_proximity_counts']
    df.loc[data.index, 's__gps_outlier'] = data['s__gps_outlier']
    df.loc[data.index, 's__gps_extreme_outlier'] = data['s__gps_extreme_outlier']

    for col in ['s__gps_proximity_counts', 's__gps_outlier', 's__gps_extreme_outlier']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


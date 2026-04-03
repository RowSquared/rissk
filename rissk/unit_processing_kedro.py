import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from pyod.models.pca import PCA
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from rissk.item_processing_kedro import get_contamination_parameter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize

logger = logging.getLogger(__name__)

def windsorize_95_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Windsorize values in all columns of the DataFrame that are above the 95th percentile.

    Args:
    - df (pd.DataFrame): Input DataFrame

    Returns:
    - pd.DataFrame: Windsorized DataFrame
    """
    df_out = df.copy()
    for column in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[column]):
            # Calculate the 95th percentile for the column
            percentile_95 = df_out[column].quantile(0.95)

            # Set values above the 95th percentile to the value at the 95th percentile
            df_out[column] = df_out[column].apply(lambda x: min(x, percentile_95))

    return df_out

# -- Note: Aggregation methods map closely to make_score_unit__* in legacy --

def aggregate_unit_score_mean(df_item_scores: pd.DataFrame, df_unit: pd.DataFrame, score_source_name: str, score_target_name: str) -> pd.DataFrame:
    """Helper purely taking interview__id mapped scores and doing mean aggregation into unit df."""
    if score_source_name not in df_item_scores.columns:
         return df_unit
         
    data = df_item_scores.groupby(['interview__id']).agg({score_source_name: 'mean'})
    df_out = df_unit.copy()
    df_out[score_target_name] = df_out['interview__id'].map(data[score_source_name])
    df_out[score_target_name] = df_out[score_target_name].fillna(0)
    return df_out

def calculate_global_score(df_unit_scores: pd.DataFrame, df_resp_scores: pd.DataFrame, score_columns: List[str], combine_resp_score: bool = True, restricted_columns: List[str] = None) -> pd.DataFrame:
    """
    Calculate the global unit risk score.
    Maps legacy `make_global_score` inside UnitDataProcessing.
    """
    df_unit = df_unit_scores.copy()
    df_unit['unit_risk_score'] = 0
    scaler = StandardScaler()
    
    # Select columns
    columns = score_columns if score_columns else []
    if restricted_columns is not None:
         columns = [col for col in columns if col not in restricted_columns]
         
    available_cols = [c for c in columns if c in df_unit.columns]
    
    if not available_cols:
        logger.warning("No score columns available to compute global risk score.")
        return df_unit

    df = df_unit[available_cols].copy()

    # Drop constant columns before StandardScaler — a constant column produces NaN after
    # z-scoring (division by zero std), which would make IForest scores meaningless and
    # MinMaxScaler produce NaN unit_risk_score for every interview.
    # This mirrors legacy's `nunique() > 1` filter in df_unit_score.
    varying_cols = [c for c in available_cols if df[c].nunique() > 1]
    if not varying_cols:
        logger.warning("All score columns are constant — cannot compute meaningful global risk score.")
        return df_unit

    df = df[varying_cols]
    df = pd.DataFrame(scaler.fit_transform(df), columns=varying_cols)
    
    model = IForest(random_state=42)
    model.fit(df.fillna(0))
    
    scaler = MinMaxScaler(feature_range=(0, 100))
    df_unit['unit_risk_score'] = model.decision_scores_

    # Windsorize
    df_unit['unit_risk_score'] = windsorize_95_percentile(df_unit[['unit_risk_score']])['unit_risk_score']

    # Scale to 0-100
    df_unit['unit_risk_score'] = scaler.fit_transform(df_unit[['unit_risk_score']])

    # Merge unit score with responsible score.
    # Only apply the multiplication when responsible_score has actual variance — if PCA
    # on the responsible-level scores couldn't run (too few enumerators or all scores
    # constant), responsible_score is all-zero, and multiplying produces a constant-zero
    # column that MinMaxScaler turns into NaN for every interview.
    if combine_resp_score and 'responsible' in df_unit.columns and df_resp_scores is not None and 'responsible_score' in df_resp_scores.columns:
        resp_score_series = df_resp_scores['responsible_score']
        if resp_score_series.nunique() > 1:
            df_resp_map = df_resp_scores.set_index('responsible')['responsible_score'].to_dict()
            df_unit['responsible_score'] = df_unit['responsible'].map(df_resp_map).fillna(0)
            df_unit['unit_risk_score'] = df_unit['unit_risk_score'] * df_unit['responsible_score']
            df_unit['unit_risk_score'] = scaler.fit_transform(df_unit[['unit_risk_score']])
        else:
            logger.warning(
                "responsible_score has no variance (likely too few enumerators or all scores constant); "
                "skipping responsible-score multiplication to preserve interview-level unit_risk_score."
            )

    return df_unit

def aggregate_item_to_unit_scores(df_unit: pd.DataFrame, df_item_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregates item-level scores up to the unit (interview) level."""
    df_out = df_unit.copy()
    
    # 1. Simple mean aggregations
    # Note: s__answer_removed is intentionally excluded here — it is scored
    # at unit level directly from paradata_full by calculate_answer_removed_unit_score
    # in calculate_unit_scores, so that items deleted from microdata are included.
    mean_scores = [
        's__answer_hour_set', 's__answer_changed',
        's__first_decimal', 's__sequence_jump'
    ]
    for score in mean_scores:
        if score in df_item_scores.columns:
            data = df_item_scores.groupby('interview__id')[score].mean()
            df_out[score] = df_out['interview__id'].map(data).fillna(0)

    # 2. Lower/Upper mean aggregations
    lower_upper_scores = [
        's__answer_selected', 's__answer_duration'
    ]
    for score_base in lower_upper_scores:
        for suffix in ['_lower', '_upper']:
            score = score_base + suffix
            if score in df_item_scores.columns:
                data = df_item_scores.groupby('interview__id')[score].mean()
                df_out[score] = df_out['interview__id'].map(data).fillna(0)

    # 3. GPS specifics (if gps scores exist)
    # s__gps is the per-interview count of GPS-type questions (sum of the item-level
    # boolean flag converted to int in calculate_gps_score), matching legacy
    # make_score_unit__gps which summed f__gps from df_item.
    gps_features = ['s__gps_proximity_counts', 's__gps_outlier', 's__gps_extreme_outlier', 's__gps']
    for score in gps_features:
        if score in df_item_scores.columns:
            data = df_item_scores.groupby('interview__id')[score].sum()
            df_out[score] = df_out['interview__id'].map(data).fillna(0)
            
    return df_out

def calculate_unit_level_scores(df_unit: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Calculate scores that are purely derived from unit-level features."""
    df = df_unit.copy()
    
    if 'f__time_changed' in df.columns:
        df['s__time_changed'] = round(df['f__time_changed'].abs() / 600)
        
    if 'f__total_duration' in df.columns:
        df['s__total_duration'] = round(df['f__total_duration'] / 300)
        
    if 'f__days_from_start' in df.columns:
        df['s__days_from_start'] = (df['f__days_from_start'] / 7).astype(int)
        
    if 'f__total_elapse' in df.columns:
        score_name = 's__total_elapse'
        df['f__total_elapse_scaled'] = round(df['f__total_elapse'] / 300)
        
        # contamination from parameters or fallback
        contamination = get_contamination_parameter(
            parameters.get('features', {}), 'f__total_elapse', method='medfilt', random_state=42
        )
            
        model = ECOD(contamination=contamination)
        valid_mask = ~df['f__total_elapse_scaled'].isnull()
        if valid_mask.sum() > 0:
            model.fit(df.loc[valid_mask, ['f__total_elapse_scaled']])
            df.loc[valid_mask, score_name] = model.predict(df.loc[valid_mask, ['f__total_elapse_scaled']])
            
            score_name1 = score_name + '_lower'
            score_name2 = score_name + '_upper'
            df[score_name1] = 0
            df[score_name2] = 0
            
            non_anomalies = df.loc[(df[score_name] == 0) & valid_mask, 'f__total_elapse_scaled']
            if not non_anomalies.empty:
                min_val = non_anomalies.min()
                max_val = non_anomalies.max()
                df.loc[valid_mask & (df['f__total_elapse_scaled'] < min_val), score_name1] = 1
                df.loc[valid_mask & (df['f__total_elapse_scaled'] > max_val), score_name2] = 1
            df.drop(columns=[score_name, 'f__total_elapse_scaled'], inplace=True, errors='ignore')

    if 'f__pause_duration' in df.columns and 'f__total_elapse' in df.columns:
        df['s__pause_duration'] = np.where(df['f__total_elapse'] != 0, 
                                           df['f__pause_duration'] / df['f__total_elapse'], 0)
        
    if 'f__pause_count' in df.columns and 'f__number_answered' in df.columns:
        df['s__pause_count'] = np.where(df['f__number_answered'] != 0, 
                                        df['f__pause_count'] / df['f__number_answered'], 0)

    if 'f__number_answered' in df.columns:
        df['s__number_answered'] = df['f__number_answered']
        
    if 'f__number_unanswered' in df.columns:
        df['s__number_unanswered'] = df['f__number_unanswered']

    return df

def aggregate_item_to_responsible_scores(df_resp: pd.DataFrame, df_item_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregates item-level scores to the responsible (enumerator) level."""
    df_out = df_resp.copy()
    if df_out.empty and 'responsible' in df_item_scores.columns:
        df_out = pd.DataFrame({'responsible': df_item_scores['responsible'].unique()})
    
    if df_out.empty:
        return df_out

    # Mean across responsible directly
    scores_double_mean = ['s__single_question', 's__multi_option_question', 's__answer_position']
    for score in scores_double_mean:
        if score in df_item_scores.columns:
            data = df_item_scores.groupby(['responsible', 'variable_name'])[score].mean().reset_index()
            data = data.groupby('responsible')[score].mean()
            if 'responsible' in df_out.columns:
                df_out[score] = df_out['responsible'].map(data).fillna(0)

    if 's__first_digit' in df_item_scores.columns:
        data = df_item_scores.groupby('responsible')['s__first_digit'].mean()
        if 'responsible' in df_out.columns:
            df_out['s__first_digit'] = df_out['responsible'].map(data).fillna(0)

    return df_out

def calculate_responsible_score(df_resp_features: pd.DataFrame, restricted_columns: List[str] = None) -> pd.DataFrame:
    """
    Calculate the global responsible (enumerator) score using PCA.
    Maps legacy `make_responsible_score`.
    """
    df_resp = df_resp_features.copy()
    if df_resp.empty or 'responsible' not in df_resp.columns:
        return df_resp
        
    scaler = StandardScaler()
    columns = [col for col in df_resp.columns if not col.startswith('responsible') and (not restricted_columns or col not in restricted_columns)]
    
    if not columns:
        df_resp['responsible_score'] = 0.0
        return df_resp

    df_grouped = df_resp.groupby('responsible')[columns].mean().reset_index()
    
    df_pca_input = df_grouped[columns].fillna(0)
    df_pca_input = df_pca_input.loc[:, df_pca_input.nunique() != 1]
    
    if df_pca_input.empty:
         df_resp['responsible_score'] = 0.0
         return df_resp
         
    df_pca_scaled = pd.DataFrame(scaler.fit_transform(df_pca_input), columns=df_pca_input.columns)
    
    model = PCA(random_state=42)
    model.fit(df_pca_scaled)
    df_grouped['responsible_score'] = model.decision_scores_
    
    df_grouped['responsible_score'] = normalize(df_grouped[['responsible_score']], norm='l1', axis=0)
    
    # Merge back to original resp mapping
    return df_resp.merge(df_grouped[['responsible', 'responsible_score']], on='responsible', how='left')

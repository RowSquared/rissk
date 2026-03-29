import os
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from rissk.unit_proccessing import *
from rissk.config import PROJ_ROOT
import hydra
# from memory_profiler import memory_usage
import warnings

warnings.simplefilter(action='ignore', category=Warning)


def manage_path(config):
    root_path = HydraConfig.get().runtime.cwd
    if config['export_path'] is not None:
        if os.path.isabs(config['export_path']) is False:
            config['export_path'] = os.path.join(root_path, config['export_path'])
        config['environment']['data']['externals'] = os.path.dirname(config['export_path'])
        for key, value in config['environment']['data'].items():
            # Check if the value is a relative path
            if not os.path.isabs(value):
                # Convert the relative path to an absolute path
                config['environment']['data'][key] = os.path.join(root_path, value)
        config['surveys'] = [os.path.basename(config['export_path'])]
    if os.path.isabs(config['output_file']) is False:

        config['output_file'] = os.path.join(root_path, config['output_file'])
    return config


@hydra.main(config_path='configuration', version_base='1.1', config_name='main.yaml')
def unit_risk_score(config: DictConfig) -> None:
    # print(OmegaConf.to_yaml(config))
    print("*" * 12)
    config = manage_path(config)

    # --- MONKEY PATCH FOR TESTING ---
    import pandas as pd
    from rissk.unit_proccessing import UnitDataProcessing
    
    # Path to your existing files
    SURVEY = "hies2024" 
    # SURVEY = "pmpmd" 
    # SURVEY = "slchbs"
    # SURVEY = "fbf house holduntitled folder"
    DATA_DIR = os.path.join(PROJ_ROOT, "rissk_kedro", "data", SURVEY, "latest", "30_PROCESSED")
    
    print(f"LOADING PARQUET FROM: {DATA_DIR}")
    
    # Load dataframes directly
    microdata = pd.read_parquet(os.path.join(DATA_DIR, "microdata.parquet"))
    paradata = pd.read_parquet(os.path.join(DATA_DIR, "paradata_processed.parquet"))

    # Manually initialize the class, skipping __init__ logic that breaks
    # We use __new__ to create instance without calling __init__
    survey_class = UnitDataProcessing.__new__(UnitDataProcessing)
    
    # Manually set attributes that __init__ would set
    survey_class.config = config
    survey_class._limit_unit = config.get('limit_unit', None)
    survey_class._allowed_features = ['f__' + k for k, v in config['features'].items() if v['use']]
    survey_class.item_level_columns = ['interview__id', 'variable_name', 'roster_level']

    # --- Prepare paradata (paradata_processed) for use as _df_paradata ---
    # process_paradata() calls fillna('') so NaN question_scope values match the isin([0, '']) filter
    # in the df_active_paradata property. Replicate that here.
    paradata.fillna('', inplace=True)

    # Normalize qnr/qnr_version -> survey_name/survey_version if the extract uses alternate column names
    alias_map = {'qnr': 'survey_name', 'qnr_version': 'survey_version'}
    for src_col, dst_col in alias_map.items():
        if src_col in paradata.columns and dst_col not in paradata.columns:
            print(f"Renaming paradata column {src_col} -> {dst_col}")
            paradata = paradata.rename(columns={src_col: dst_col})

    # Ensure survey_name / survey_version exist (fallback if still missing)
    for col in ['survey_name', 'survey_version']:
        if col not in paradata.columns:
            print(f"Adding missing column to paradata: {col}")
            paradata[col] = SURVEY if col == 'survey_name' else 'latest'

    # _df_paradata = processed paradata (equivalent to process_paradata() output).
    # df_active_paradata property will derive the active subset from this automatically.
    survey_class._df_paradata = paradata
    
    # BYPASS make_df_item call in __init__ and do it manually
    print("Building Items (Legacy)...")
    survey_class._df_item = survey_class.make_df_item(microdata)
    
    print("Building Units (Legacy)...")
    survey_class._df_unit = survey_class.make_df_unit()
    
    print("Building Responsible...")
    survey_class._df_resp = survey_class.make_df_responsible()
    
    # Numeric mask setup (copied from __init__)
    survey_class.numeric_question_mask = (
                (survey_class._df_item["qtype"] == 'NumericQuestion') &
                (survey_class._df_item['value'] != '') &
                (~pd.isnull(survey_class._df_item['value'])) &
                (survey_class._df_item['value'] != -999999999)
    )
    
    # Initialize score columns to None
    survey_class._score_columns = None

    # Now standard flow
    try:
        # Calculate features (accessing properties triggers calculations)
        print("Calculating Item Features...")
        _ = survey_class.df_item 
        
        print(f"saving item features to {DATA_DIR}/item_features_legacy.parquet")
        survey_class._df_item.to_parquet(os.path.join(DATA_DIR, "item_features_legacy.parquet"))

        print("Calculating Unit Features...")
        _ = survey_class.df_unit
        
        print(f"saving unit features to {DATA_DIR}/unit_features_legacy.parquet")
        survey_class._df_unit.to_parquet(os.path.join(DATA_DIR, "unit_features_legacy.parquet"))

        print("Calculating Global Legacy Risk Scores...")
        survey_class.make_global_score()

        # Persist final unit risk output from legacy code path.
        unit_risk_cols = ['interview__id', 'responsible', 'unit_risk_score']
        unit_risk_df = survey_class._df_unit[unit_risk_cols].copy()
        unit_risk_df['unit_risk_score'] = unit_risk_df['unit_risk_score'].round(2)
        unit_risk_df.sort_values('unit_risk_score', inplace=True)
        unit_risk_csv = os.path.join(DATA_DIR, "unit_risk_score_legacy.csv")
        unit_risk_parquet = os.path.join(DATA_DIR, "unit_risk_score_legacy.parquet")
        unit_risk_df.to_csv(unit_risk_csv, index=False)
        unit_risk_df.to_parquet(unit_risk_parquet, index=False)

        # Build a merged score table to inspect item/unit/responsible scoring columns together.
        resp_score_cols = [c for c in survey_class._df_resp.columns if c.startswith('s__')]
        resp_view_cols = ['responsible', 'responsible_score'] + resp_score_cols
        df_scores = survey_class._df_unit.merge(
            survey_class._df_resp[resp_view_cols],
            on='responsible',
            how='left',
        )
        score_cols = [c for c in df_scores.columns if c.startswith('s__')]
        id_cols = [
            c for c in ['interview__id', 'responsible', 'survey_name', 'survey_version']
            if c in df_scores.columns
        ]
        final_cols = id_cols + ['unit_risk_score', 'responsible_score'] + sorted(score_cols)
        final_cols = [c for c in final_cols if c in df_scores.columns]
        df_scores = df_scores[final_cols]

        scores_csv = os.path.join(DATA_DIR, "scores_table_legacy.csv")
        scores_parquet = os.path.join(DATA_DIR, "scores_table_legacy.parquet")
        df_scores.to_csv(scores_csv, index=False)
        df_scores.to_parquet(scores_parquet, index=False)
        print(f"saved legacy unit risk to {unit_risk_csv}")
        print(f"saved legacy score table to {scores_csv}")
        
        print("DONE. Legacy test data generated.")
        return # Stop here
        
    except ValueError as e:
        print(f"An error occurred: {e}")

    # --- END MONKEY PATCH ---

    try:
        survey_class = UnitDataProcessing(config)
        df_item = survey_class.df_item
        df_unit = survey_class.df_unit
        survey_class.make_global_score()
        survey_class.save()
    except ValueError as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    unit_risk_score()
    # mem_usage = memory_usage(unit_risk_score)
    # print(f"Memory usage (in MB): {max(mem_usage)}")

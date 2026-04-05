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
    # Loads Kedro feature-creation pipeline outputs (item_features, unit_features,
    # removed_answers) and runs only the legacy scoring logic on top of them so
    # the resulting scores can be compared to the Kedro scoring pipeline outputs.
    import pandas as pd
    from rissk.unit_proccessing import UnitDataProcessing

    SURVEY = "hies2024"
    # SURVEY = "pmpmd"
    # SURVEY = "slchbs"
    # SURVEY = "fbf house holduntitled folder"
    DATA_DIR = os.path.join(PROJ_ROOT, "rissk_kedro", "data", SURVEY, "latest", "30_PROCESSED")
    SCORE_DIR = os.path.join(PROJ_ROOT, "rissk_kedro", "data", SURVEY, "latest", "40_SCORED")

    print(f"LOADING KEDRO FEATURE OUTPUTS FROM: {DATA_DIR}")

    # Load Kedro feature-creation pipeline outputs
    df_item_kedro     = pd.read_parquet(os.path.join(DATA_DIR, "item_features.parquet"))
    df_unit_kedro     = pd.read_parquet(os.path.join(DATA_DIR, "unit_features.parquet"))
    df_removed_kedro  = pd.read_parquet(os.path.join(DATA_DIR, "removed_answers.parquet"))

    # Get unique questionnaires present in the feature tables (mirrors pipeline_registry per-qnr loop)
    qnr_names = df_unit_kedro['qnr'].dropna().unique().tolist()
    print(f"Found {len(qnr_names)} questionnaire(s): {qnr_names}")

    all_item_scores: list = []
    all_unit_risk_dfs: list = []
    all_df_scores: list = []

    for qnr_name in qnr_names:
        print(f"\n--- Processing questionnaire: {qnr_name} ---")

        # Filter each feature table to this questionnaire only (mirrors make_qnr_filter)
        df_item_qnr = df_item_kedro[df_item_kedro['qnr'] == qnr_name].copy()
        df_unit_qnr = df_unit_kedro[df_unit_kedro['qnr'] == qnr_name].copy()
        if df_removed_kedro is not None and not df_removed_kedro.empty:
            if 'qnr' in df_removed_kedro.columns:
                df_removed_qnr = df_removed_kedro[df_removed_kedro['qnr'] == qnr_name].copy()
            else:
                # fallback: removed_answers pre-dates the qnr column addition
                valid_ids = set(df_unit_qnr['interview__id'])
                df_removed_qnr = df_removed_kedro[df_removed_kedro['interview__id'].isin(valid_ids)].copy()
        else:
            df_removed_qnr = pd.DataFrame()

        if df_unit_qnr.empty:
            print(f"  No units found for {qnr_name}, skipping.")
            continue

        # Manually initialize the class without calling __init__
        survey_class = UnitDataProcessing.__new__(UnitDataProcessing)
        survey_class.config = config
        survey_class._limit_unit = config.get('limit_unit', None)
        survey_class._allowed_features = ['f__' + k for k, v in config['features'].items() if v['use']]
        survey_class.item_level_columns = ['interview__id', 'variable_name', 'roster_level']

        # Assign filtered feature tables; strip any pre-existing s__* columns so that
        # make_global_score starts from a clean slate.
        # reset_index(drop=True) is critical: boolean-filter slices of the parquet
        # retain the original row positions (e.g. pmpmd_household rows may start at
        # index 25 if pmpmd_individual occupies rows 0..24). make_global_score merges
        # _df_unit with _df_resp and then assigns the result back by label — any row
        # whose index label exceeds len(merged_df)-1 gets NaN, producing blank unit_risk_scores.
        survey_class._df_item = df_item_qnr.drop(
            columns=[c for c in df_item_qnr.columns if c.startswith('s__')]
        ).reset_index(drop=True)
        survey_class._df_unit = df_unit_qnr.drop(
            columns=[c for c in df_unit_qnr.columns if c.startswith('s__')]
        ).reset_index(drop=True)
        # df_unit_score property requires survey_name/survey_version; rename from Kedro column names
        survey_class._df_unit.rename(columns={'qnr': 'survey_name', 'qnr_version': 'survey_version'}, inplace=True)
        if 'survey_name' not in survey_class._df_unit.columns:
            survey_class._df_unit['survey_name'] = qnr_name
        if 'survey_version' not in survey_class._df_unit.columns:
            survey_class._df_unit['survey_version'] = 'latest'

        # Build _df_resp from unique responsibles present in this questionnaire's unit features
        survey_class._df_resp = (
            df_unit_qnr[['responsible']]
            .drop_duplicates()
            .loc[lambda d: (d['responsible'] != '') & d['responsible'].notna()]
            .reset_index(drop=True)
            .copy()
        )

        # Numeric mask needed by several scoring methods accessed via self.df_item
        survey_class.numeric_question_mask = (
            (survey_class._df_item["qtype"] == 'NumericQuestion') &
            (survey_class._df_item['value'] != '') &
            (~pd.isnull(survey_class._df_item['value'])) &
            (survey_class._df_item['value'] != -999999999)
        )

        survey_class._score_columns = None

        # Patch get_feature_item__answer_removed so that make_score__answer_removed
        # uses the Kedro-built removed_answers table instead of reading self.df_paradata.
        # Default argument captures df_removed_qnr at loop iteration time.
        survey_class.get_feature_item__answer_removed = lambda feature_name, _r=df_removed_qnr: _r.copy()

        try:
            print(f"  Calculating Legacy Risk Scores for {qnr_name}...")

            # Populate all s__* columns on _df_unit/_df_resp first, then sanitise before
            # StandardScaler runs. Division-based scores (e.g. s__pause_duration =
            # f__pause_duration / f__total_elapse) can produce inf when the denominator is 0.
            _ = survey_class.df_unit_score
            s_cols = [c for c in survey_class._df_unit.columns if c.startswith('s__')]
            survey_class._df_unit[s_cols] = survey_class._df_unit[s_cols].replace(
                [np.inf, -np.inf], np.nan
            )

            # Recompute _score_columns on sanitised data so make_global_score sees the
            # correct set. For small surveys all scores can be constant/all-NaN after
            # sanitisation, which would give StandardScaler an empty DataFrame.
            score_cols_all = [c for c in survey_class._df_unit.columns if c.startswith('s__')]
            survey_class._score_columns = (
                survey_class._df_unit[score_cols_all]
                .columns[survey_class._df_unit[score_cols_all].nunique() > 1]
                .tolist()
            )
            if not survey_class._score_columns:
                print(f"  No score columns with sufficient variance for {qnr_name} — skipping global score.")
                continue

            # Determine whether the responsible-level score has enough variance to run PCA.
            _restricted = survey_class._score_columns
            _resp_candidates = [
                c for c in survey_class._df_resp.columns
                if not c.startswith('responsible') and c not in _restricted
            ]
            _resp_has_variance = (
                not survey_class._df_resp[_resp_candidates].fillna(0).loc[
                    :, survey_class._df_resp[_resp_candidates].fillna(0).nunique() != 1
                ].empty
                if _resp_candidates else False
            )

            survey_class.make_global_score(combine_resp_score=_resp_has_variance)

            # Build item-level score table (equivalent to Kedro calculate_item_scores output).
            # answer_removed is excluded here matching Kedro behaviour (scored at unit level only).
            # GPS is excluded due to its pivoted shape (already a WARNING in make_global_score).
            print(f"  Collecting item-level scores for {qnr_name}...")
            id_cols = [c for c in ['interview__id', 'variable_name', 'roster_level', 'index_col']
                       if c in survey_class._df_item.columns]
            df_item_scores = survey_class._df_item[id_cols].copy()
            merge_key = 'index_col' if 'index_col' in df_item_scores.columns \
                else ['interview__id', 'variable_name', 'roster_level']
            merge_cols = [merge_key] if isinstance(merge_key, str) else merge_key

            item_score_methods = [
                ('make_score__answer_hour_set',      ['s__answer_hour_set']),
                ('make_score__sequence_jump',         ['s__sequence_jump']),
                ('make_score__first_decimal',         ['s__first_decimal']),
                ('make_score__answer_changed',        ['s__answer_changed']),
                ('make_score__answer_position',       ['s__answer_position']),
                ('make_score__answer_selected',       ['s__answer_selected_lower', 's__answer_selected_upper']),
                ('make_score__answer_duration',       ['s__answer_duration_lower', 's__answer_duration_upper']),
                ('make_score__single_question',       ['s__single_question']),
                ('make_score__multi_option_question', ['s__multi_option_question']),
                ('make_score__first_digit',           ['s__first_digit']),
            ]
            for method_name, score_cols in item_score_methods:
                try:
                    result = getattr(survey_class, method_name)()
                    available = [c for c in score_cols if c in result.columns]
                    if not available:
                        continue
                    result_slim = result[merge_cols + available].drop_duplicates(subset=merge_cols)
                    df_item_scores = df_item_scores.merge(result_slim, on=merge_key, how='left')
                except Exception as e:
                    print(f"  WARNING: item score {score_cols}: {e}")

            all_item_scores.append(df_item_scores)

            # Collect unit risk scores for this questionnaire
            unit_risk_cols = ['interview__id', 'responsible', 'unit_risk_score']
            unit_risk_df = survey_class._df_unit[unit_risk_cols].copy()
            unit_risk_df['unit_risk_score'] = unit_risk_df['unit_risk_score'].round(2)
            all_unit_risk_dfs.append(unit_risk_df)

            # Build merged score table (unit + responsible scores) for this questionnaire
            resp_score_cols = [c for c in survey_class._df_resp.columns if c.startswith('s__')]
            resp_id_cols = ['responsible']
            if 'responsible_score' in survey_class._df_resp.columns:
                resp_id_cols.append('responsible_score')
            resp_view_cols = resp_id_cols + resp_score_cols
            df_scores_qnr = survey_class._df_unit.merge(
                survey_class._df_resp[resp_view_cols], on='responsible', how='left',
            )
            score_cols_final = [c for c in df_scores_qnr.columns if c.startswith('s__')]
            id_cols_final = [c for c in ['interview__id', 'responsible', 'survey_name', 'survey_version']
                             if c in df_scores_qnr.columns]
            final_cols = id_cols_final + ['unit_risk_score', 'responsible_score'] + sorted(score_cols_final)
            df_scores_qnr = df_scores_qnr[[c for c in final_cols if c in df_scores_qnr.columns]]
            all_df_scores.append(df_scores_qnr)

        except ValueError as e:
            print(f"  ERROR in {qnr_name}: {e}")
            continue

    # --- Merge per-questionnaire results (mirrors merge_pipeline in pipeline_registry) ---
    if not all_item_scores:
        print("No questionnaire produced results. Exiting.")
        return

    df_item_scores_all = pd.concat(all_item_scores, ignore_index=True)
    # Normalise any object-typed columns that became mixed after concat
    for col in df_item_scores_all.columns:
        if df_item_scores_all[col].dtype == object:
            try:
                df_item_scores_all[col] = df_item_scores_all[col].astype(float)
            except (ValueError, TypeError):
                pass
    item_scores_parquet = os.path.join(SCORE_DIR, "item_scores_legacy.parquet")
    df_item_scores_all.to_parquet(item_scores_parquet, index=False)
    print(f"Saved legacy item scores to {item_scores_parquet}")

    unit_risk_df_all = pd.concat(all_unit_risk_dfs, ignore_index=True)
    unit_risk_df_all.sort_values('unit_risk_score', inplace=True)
    unit_risk_csv     = os.path.join(SCORE_DIR, "unit_risk_score_legacy.csv")
    unit_risk_parquet = os.path.join(SCORE_DIR, "unit_risk_score_legacy.parquet")
    unit_risk_df_all.to_csv(unit_risk_csv, index=False)
    unit_risk_df_all.to_parquet(unit_risk_parquet, index=False)

    df_scores_all = pd.concat(all_df_scores, ignore_index=True)
    # After concat across questionnaires, boolean/int columns can become object dtype.
    # Cast any remaining object-typed s__* columns to float so pyarrow can write parquet.
    for col in df_scores_all.columns:
        if df_scores_all[col].dtype == object:
            try:
                df_scores_all[col] = df_scores_all[col].astype(float)
            except (ValueError, TypeError):
                pass  # leave non-numeric object columns as-is
    scores_csv     = os.path.join(SCORE_DIR, "scores_table_legacy.csv")
    scores_parquet = os.path.join(SCORE_DIR, "scores_table_legacy.parquet")
    df_scores_all.to_csv(scores_csv, index=False)
    df_scores_all.to_parquet(scores_parquet, index=False)
    print(f"Saved legacy unit risk to {unit_risk_csv}")
    print(f"Saved legacy score table to {scores_csv}")

    print("DONE. Legacy scores from Kedro features generated.")
    return  # Stop here — do not fall through to the regular UnitDataProcessing block

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

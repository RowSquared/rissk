from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional
import re
import os
import zipfile
import json  # Added json import
import pandas as pd  # Added pandas import
import numpy as np   # Added numpy import

from loguru import logger

from rissk.utils.file_process_utils import (
    get_file_parts, 
    transform_multi,
    set_qnr_version, 
    normalize_column_name,
    process_json_structure, 
    get_categories,
    update_df_categories,
    parse_filename
)


def extract_zip(file_source_path: Path, file_dest_path: Path, password: Optional[str] = None):
    """Memory-efficient recursive extraction for Python 3.13."""
    current_pwd = password or os.getenv('PASSWORD')
    pwd_bytes = current_pwd.encode() if current_pwd else None

    # Ensure destination exists
    file_dest_path.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(file_source_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                target_path = file_dest_path / file_info.filename
                
                # Prevent directory traversal vulnerability
                if not str(target_path.resolve()).startswith(str(file_dest_path.resolve())):
                    logger.warning(f"Skipping extraction of {file_info.filename}: path traversal attempt")
                    continue
                
                if file_info.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue

                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                if target_path.exists():
                     # Optional: Skip already extracted files or overwrite
                     pass

                with zip_ref.open(file_info, pwd=pwd_bytes) as source, \
                     open(target_path, "wb") as target:
                    target.write(source.read())

                if target_path.suffix.lower() == '.zip':
                    nested_dest = target_path.with_suffix('')
                    extract_zip(target_path, nested_dest, password=current_pwd)
        
        logger.info(f"Extracted: {file_source_path.name}")
    except Exception as e:
        logger.error(f"Failed {file_source_path}: {e}")


def _get_partition_path(partition_id: str, loader: Any) -> Optional[Path]:
    """
    Robustly resolve partition path from a Kedro partition loader.
    Compatible with Kedro 0.18+ and standard partition loaders.
    """
    # 1. Try to get path from loader if it's a bound method (most datasets)
    dataset = getattr(loader, "__self__", None)
    if dataset:
        for attr in ("_filepath", "filepath", "path", "_path"):
            path = getattr(dataset, attr, None)
            if path:
                return Path(path)

    # 2. Try inspection for closures (legacy fallback)
    try:
        closure = getattr(loader, "__closure__", None)
        if closure:
            for cell in closure:
                content = cell.cell_contents
                for attr in ("_filepath", "filepath", "path", "_path"):
                    path = getattr(content, attr, None)
                    if path:
                        return Path(path)
    except Exception:
        pass
    
    # 3. Last resort: Assume partition_id is relative to current working directory
    # (Unlikely in Kedro context but safe fallback structure wise if ID is path-like)
    candidate = Path(partition_id)
    if candidate.exists():
        return candidate
        
    return None


def extract_all_zip_files(partitions: dict[str, Any], zip_password: str = None) -> None:
    """
    Extract all zip files referenced by Kedro partition IDs.
    Recursively extracts nested zips.
    """
    if not partitions:
        logger.warning("No partitions found for zip extraction")
        return

    # Collect source zips
    zip_paths: list[Path] = []
    
    for partition_id, loader in partitions.items():
        # Partition keys are typically relative paths
        # We need the absolute path to the zip file
        
        # Only process items that look like zips
        if not str(partition_id).lower().endswith(".zip"):
            continue
            
        zip_path = _get_partition_path(partition_id, loader)
        
        if zip_path and zip_path.exists():
            zip_paths.append(zip_path)
        else:
            logger.warning(f"Could not resolve path for partition: {partition_id}")

    logger.info(f"Found {len(zip_paths)} top-level zip files to extract")

    for zip_path in zip_paths:
        destination = zip_path.with_suffix("")
        extract_zip(zip_path, destination, password=zip_password)


def filter_matching_folders(partitions: dict[str, Any], questionnaires: list[dict]) -> list[Path]:
    """
    Return extracted folder paths matching questionnaire/version patterns.
    Iterates over extracted folders (datasets) to find matches.
    """
    if not partitions:
        logger.warning("No partitions found while filtering extracted folders")
        return []

    matching_folders: list[Path] = []
    seen_paths = set()

    # Pre-compile patterns
    patterns = []
    for q in questionnaires:
        name = q.get("name")
        versions = q.get("VERSION", [])
        version_pattern = "|".join(map(str, versions))
        # Matches: NAME_VERSION_... (e.g. slbhies_listing_6_Paradata_All)
        patterns.append(re.compile(rf"^{name}_({version_pattern})_.*"))

    logger.info(f"Scanning {len(partitions)} folder partitions against {len(patterns)} patterns")

    for partition_id, loader in partitions.items():
        partition_path_obj = Path(partition_id)
        folder_name = partition_path_obj.name
        
        # Check against patterns
        is_match = False
        for pattern in patterns:
            if pattern.match(folder_name):
                is_match = True
                break
        
        if not is_match:
            continue

        # Resolved path
        folder_path = _get_partition_path(partition_id, loader)
        
        if folder_path and folder_path.is_dir():
            folder_str = str(folder_path.resolve())
            if folder_str not in seen_paths:
                seen_paths.add(folder_str)
                matching_folders.append(folder_path)

    logger.info(f"Found {len(matching_folders)} matching folders")
    return matching_folders


# --- Legacy Functions Migrated from import_utils.py ---

def get_survey_info(survey_files: list[Path]) -> dict[str, dict[str, dict[str, Path]]]:
    """
    Organizes survey files into a structured dictionary.
    
    Structure:
    {
        'questionnaire_name': {
            'qnr_version_string': {
                'file_format': Path(/path/to/folder)
            }
        }
    }
    """
    survey_info = {}

    for survey_path in survey_files:
        filename = survey_path.name
        try:
            questionnaire, version, file_format, interview_status = get_file_parts(filename)
        except ValueError as e:
            logger.warning(f"Skipping {filename}: {e}")
            continue
            
        qnr_version = f"{questionnaire}_{str(version)}"  

        survey_info.setdefault(questionnaire, {})
        survey_info[questionnaire].setdefault(qnr_version, {})
        survey_info[questionnaire][qnr_version][file_format] = survey_path
        
    return survey_info


def read_json_questionnaire(survey_path: Path, questionnaire_path: Optional[Path] = None) -> dict:
    """Reads the questionnaire JSON definition."""
    if questionnaire_path is None:
        file_path = survey_path / 'Questionnaire' / 'content' / 'document.json'
    else:
        # If explicit questionnaire_path is given (rare case in current pipeline usage)
        # We need check if it points to a specific file or directory
        # This part assumes structure compatible with get_questionnaire_map from legacy code
        # simplified here for clarity/robustness:
        if questionnaire_path.is_file():
             file_path = questionnaire_path
        else:
            # Fallback logic mirroring legacy get_questionnaire_id/map behavior if needed
             # For now, simplistic implementation assuming standard export structure
             file_path = survey_path / 'Questionnaire' / 'content' / 'document.json'

    if not file_path.exists():
        logger.warning(f"Questionnaire document not found at {file_path}")
        return None

    with file_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def get_questionnaire(data_path: Path, questionnaire_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads and processes a questionnaire from a JSON file located at the specified path.
    Also handles categorization of data.
    """
    q_data = read_json_questionnaire(data_path, questionnaire_path=questionnaire_path)

    qnr_df = pd.DataFrame()

    if q_data is not None:
        question_data = []
        question_counter = 0

        # process_json_structure modifies question_data list in-place
        process_json_structure(q_data.get("Children", []), "", question_counter, question_data)

        if question_data:
            qnr_df = pd.DataFrame(question_data)
            
            # Type-safe transformations
            qnr_df['answer_sequence'] = qnr_df['Answers'].apply(
                lambda x: [int(item['AnswerValue']) for item in x] if x else np.nan
            )
            qnr_df['n_answers'] = qnr_df['Answers'].apply(lambda x: len(x) if x else np.nan)
            qnr_df['is_linked'] = (qnr_df['LinkedToRosterId'].notna()) | (qnr_df['LinkedToQuestionId'].notna())
            
            if 'parents' in qnr_df.columns:
                qnr_df['parents'] = qnr_df['parents'].str.lstrip(' > ')
                split_columns = qnr_df['parents'].str.split(' > ', expand=True)
                split_columns.columns = [f"parent_{i + 1}" for i in range(split_columns.shape[1])]
                qnr_df = pd.concat([qnr_df, split_columns], axis=1)

            if 'QuestionScope' in qnr_df.columns:
                qmask = qnr_df['QuestionScope'] == 0
                qnr_df['question_sequence'] = qmask.cumsum()
                qnr_df.loc[~qmask, 'question_sequence'] = None
        
    categories_path = data_path / 'Questionnaire' / 'content' / 'Categories'

    if categories_path.exists():
        categories = get_categories(categories_path)
        if not qnr_df.empty:
            qnr_df = qnr_df.apply(lambda row: update_df_categories(row, categories), axis=1)

    if not qnr_df.empty:
        qnr_df.reset_index(drop=True, inplace=True)
        # Normalize columns
        qnr_df.columns = [normalize_column_name(c) for c in qnr_df.columns]
        
        try:
            parts = parse_filename(data_path.name)
            # parts is a list: [questionnaire, version, format, status]
            # set_qnr_version expects (df, questionaire_name, qnr_version)
            qnr_df = set_qnr_version(qnr_df, parts[0], parts[1]) # parts[0]=name, parts[1]=version
        except ValueError:
            logger.warning(f"Could not parse filename '{data_path.name}' for version info")

    return qnr_df


def read_paradata(survey_path: Path, delimiter='\t') -> pd.DataFrame:
    file_path = survey_path / 'paradata.tab'
    if not file_path.exists():
         raise FileNotFoundError(f"Paradata file not found: {file_path}")
         
    with file_path.open('r', encoding='utf-8') as f:
        # low_memory=False to avoid DtypeWarnings on large files, standard pandas practice
        df = pd.read_csv(f, delimiter=delimiter, low_memory=False)
    return df


def get_paradata(data_path: Path, df_questionnaires: pd.DataFrame) -> pd.DataFrame:
    """
    Loads and processes a paradata file from the provided path and merges it with the questionnaire dataframe.
    """
    try:
        df_para = read_paradata(data_path, delimiter='\t')
    except Exception as e:
        logger.error(f"Error reading paradata from {data_path}: {e}")
        return pd.DataFrame()

    if 'parameters' in df_para.columns:
        # split the parameter column
        # Using n=1 to limit splits is correct
        # Check if expand=True returns intended shape
        split_param = df_para['parameters'].str.split(r'\|\|', n=1, expand=True)
        if split_param.shape[1] == 2:
             df_para['param'] = split_param[0]
             df_para['answer'] = split_param[1]
        else:
             df_para['param'] = df_para['parameters']
             df_para['answer'] = None
             
        if 'answer' in df_para.columns and df_para['answer'].notna().any():     
            split_answer = df_para['answer'].str.rsplit(r'||', n=1, expand=True)
            if split_answer.shape[1] == 2:
                 df_para['answer'] = split_answer[0]
                 df_para['roster_level'] = split_answer[1]
            else:
                 df_para['roster_level'] = None # Or empty string

    return df_para


def get_microdata_file_list(data_path: Path) -> list[str]:
    """
    Get a list of microdata files in the specified directory.
    """
    excluded_prefixes = ('interview__', 'assignment__')
    excluded_files = {'paradata.tab'}
    valid_extensions = {'.dta', '.tab'}

    file_names = []
    if data_path.exists():
        for file in data_path.iterdir():
            if file.is_file() and file.suffix in valid_extensions:
                if file.name not in excluded_files and not file.name.startswith(excluded_prefixes):
                    file_names.append(file.name)
    return file_names


def read_microdata_file(data_path: Path, file_name: str) -> pd.DataFrame:
    file_path = data_path / file_name
    
    if file_path.suffix == '.dta':
        try:
            # Using 'with' open ensures file handle closure
            with file_path.open('rb') as f:
                 # convert_categoricals=False matches legacy beahvior
                df = pd.read_stata(f, convert_categoricals=False, convert_missing=True)
            
            # Vectorized replacement is faster
            # Replace '.a' Stata missing value with -999999999
            # Replace '.' Stata missing value with NaN
            # Use strict type checking or conversion to string if mixed
            
            # Safety: ensure we don't fail if column is all numeric types (no '.a')
            # convert to object if needed? usually .dta loads with correct types or object if strings exist
            
            # Legacy logic: df.astype(str) != '.a' -> expensive full copy?
            # Better: replace specific values
            df.replace({'.a': -999999999, '.': np.nan}, inplace=True)
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return pd.DataFrame()
    else: # .tab file
        try:
             with file_path.open('r', encoding='utf-8') as f:
                df = pd.read_csv(f, delimiter='\t', low_memory=False)
        except Exception as e:
            logger.error(f"Error reading csv {file_path}: {e}")
            return pd.DataFrame()
            
    return df


def get_microdata(data_path: Path, df_questionnaires: pd.DataFrame) -> pd.DataFrame:
    drop_list = {'interview__key', 'sssys_irnd', 'has__errors', 'interview__status', 'assignment__id'}

    file_names = get_microdata_file_list(data_path)
    
    # Pre-calculate masks outside loop
    multi_unlinked_vars = []
    multi_linked_vars = []
    list_vars = []
    gps_vars = []

    if not df_questionnaires.empty:
        # Use boolean indexing
        unlinked_mask = (df_questionnaires["qtype"] == 'MultyOptionsQuestion') & (df_questionnaires['is_linked'] == False)
        linked_mask = (df_questionnaires["qtype"] == 'MultyOptionsQuestion') & (df_questionnaires['is_linked'] == True)
        list_mask = (df_questionnaires["qtype"] == 'TextListQuestion')
        gps_mask = (df_questionnaires["qtype"] == 'GpsCoordinateQuestion')

        multi_unlinked_vars = df_questionnaires.loc[unlinked_mask, 'variable_name'].tolist()
        multi_linked_vars = df_questionnaires.loc[linked_mask, 'variable_name'].tolist()
        list_vars = df_questionnaires.loc[list_mask, 'variable_name'].tolist()
        gps_vars = df_questionnaires.loc[gps_mask, 'variable_name'].tolist()

    all_dfs = []
    for file_name in file_names:
        df = read_microdata_file(data_path, file_name)
        if df.empty:
            continue
            
        #Efficient drop
        cols_to_drop = [col for col in drop_list if col in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        if not df_questionnaires.empty:
            df = transform_multi(df, multi_unlinked_vars, 'unlinked')
            df = transform_multi(df, multi_linked_vars, 'linked')
            df = transform_multi(df, list_vars, 'list')
            df = transform_multi(df, gps_vars, 'gps')

        # Handle roster IDs
        roster_ids = [col for col in df.columns if col.endswith("__id") and col != "interview__id"]
        if roster_ids:
            # Vectorized string join is harder in pandas, apply is okay here
            df['roster_level'] = df[roster_ids].astype(str).agg(','.join, axis=1)
            df.drop(columns=roster_ids, inplace=True)
        else:
            df['roster_level'] = ''

        id_vars = ['interview__id', 'roster_level']
        value_vars = [col for col in df.columns if col not in id_vars]
        
        if not value_vars:
            continue
            
        df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='variable', value_name='value')
        df_long['filename'] = file_name
        all_dfs.append(df_long)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

    # Filter invalid values
    # Optimized filter:
    # Check for empty string or NaN. Note: 'value' column is mixed type probably.
    # Convert 'value' to string could simplify emptiness check but be careful with NaN
    
    # Vectorized check is faster than apply
    # combined_df['value'] is likely object type
    
    # is_valid logic from legacy: 
    # if list: return True
    # if string/other: value != '' and notna(value)
    
    # Since we can't easily vectorize types check mixed with lists in pandas, use apply only if needed
    # But usually transform_multi returns lists for some columns.
    
    def is_valid_fast(val):
        if val is None: return False
        if isinstance(val, (list, tuple)): return True # Not empty list check? Legacy said 'return True' commented 'bool(value)'
        if val == '': return False
        try:
             if pd.isna(val): return False
        except:
             pass # list not hashable for isna sometimes?
        return True

    combined_df = combined_df[combined_df['value'].apply(is_valid_fast)]

    try:
        parts = parse_filename(data_path.name)
        questionaire_name = parts[0]
        qnr_version = parts[1]
        combined_df = set_qnr_version(combined_df, questionaire_name, qnr_version)
    except ValueError:
        logger.warning(f"Could not set version for {data_path.name}")

    if not df_questionnaires.empty:
        # Merge setup
        roster_columns = [c for c in combined_df.columns if '__id' in c and c != 'interview__id']
        
        # Ensure join keys have matching types
        # variable, qnr, qnr_version are strings/objects
        
        merge_on_left = ['variable', 'qnr', 'qnr_version']
        merge_on_right = ['variable_name', 'qnr', 'qnr_version']
        
        combined_df = combined_df.merge(
            df_questionnaires, 
            how='left',
            left_on=merge_on_left,
            right_on=merge_on_right
        )
        
        sort_cols = ['interview__id']
        if 'qnr_seq' in combined_df.columns:
            sort_cols.append('qnr_seq')
        sort_cols.extend(roster_columns)
        
        # Safe sort (ignore missing cols)
        actual_sort_cols = [c for c in sort_cols if c in combined_df.columns]
        combined_df.sort_values(actual_sort_cols, inplace=True)

    combined_df.reset_index(drop=True, inplace=True)
    combined_df.columns = [normalize_column_name(c) for c in combined_df.columns]
    combined_df['value'] = combined_df['value'].astype(str)
    
    return combined_df

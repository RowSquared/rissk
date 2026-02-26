from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Dict, List
import re
import os
import zipfile
import shutil
import json  # Added json import
import pandas as pd  # Added pandas import
import numpy as np   # Added numpy import

from loguru import logger

from rissk.utils.file_process_utils_kedro import (
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
    """Memory-efficient recursive extraction."""
    pwd_bytes = password.encode() if password else None
    file_dest_path.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(file_source_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                target_path = (file_dest_path / file_info.filename).resolve()
                
                # Security: Prevent ZipSlip/Path Traversal
                if not str(target_path).startswith(str(file_dest_path.resolve())):
                    continue
                
                if file_info.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue

                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Stream content to file to keep memory usage low
                with zip_ref.open(file_info, pwd=pwd_bytes) as source, \
                     open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

                # Recursive call for nested zips
                if target_path.suffix.lower() == '.zip':
                    extract_zip(target_path, target_path.with_suffix(''), password=password)
                    
    except Exception as e:
        logger.error(f"Failed to extract {file_source_path.name}: {e}")


def filter_matching_folders(partitions: Dict[str, Callable[[], Path]], questionnaires: List[Dict]) -> List[Path]:
    """
    Filters partition paths to return only directories that match 
    specific questionnaire name and version patterns.
    """
    if not partitions:
        logger.warning("No partitions found while filtering extracted folders.")
        return []

    # 1. Pre-compile patterns for efficiency
    # We use \b or strict string termination to ensure version 1 doesn't match 10
    patterns = []
    for q in questionnaires:
        name = q.get("name")
        versions = q.get("VERSION", [])
        if not name or not versions:
            continue
            
        version_pattern = "|".join(map(str, versions))
        # Pattern: Matches start of string, the name, an underscore, 
        # one of the versions, and then an underscore or end of string.
        # Example: ^slbhies_listing_(1|2|6)_.*
        regex = re.compile(rf"^{re.escape(name)}_({version_pattern})_.*")
        patterns.append(regex)

    matching_folders: List[Path] = []
    seen_paths = set()

    # 2. Iterate and validate
    # FIX: Use partition keys to strictly identify the top-level folder relative to the root.
    # Keys in PartitionedDataset are relative paths like "SurveyFolder/Sub/File.ext".
    # We only check the first component ("SurveyFolder") against the regex.
    
    for partition_id, loader in partitions.items():
        try:
            # partition_id is the relative path (e.g. "folder/sub/file.txt")
            # We normalize it to a Path object to handle OS separators safely
            relative_path = Path(partition_id)
            
            # We expect at least a folder and a file (parts > 1)
            # If the zip extracted to flat files at root, this checks prevents errors.
            if len(relative_path.parts) < 2:
                continue

            # The top-level folder name is the first part of the relative path
            top_level_name = relative_path.parts[0]
            
            # Check if this top-level folder matches our patterns
            is_match = False
            for pattern in patterns:
                if pattern.match(top_level_name):
                    is_match = True
                    break
            
            if is_match:
                # Calculate the absolute path of the top-level folder
                # We do this by taking the file's full path and stripping the
                # sub-directories indicated by the relative path key.
                file_path = loader()
                
                # We need to go up N levels where N = number of parts in relative path - 1
                # Example: Key="A/B/file" (3 parts). Path=".../A/B/file". 
                # We want ".../A". We need to go up 2 levels (file->B, B->A).
                levels_up = len(relative_path.parts) - 1
                
                # parents[0] is the directory containing the file.
                # parents[levels_up-1] is the directory we want.
                # Path.parents sequence: [parent, parent.parent, ...]
                # Index 0 is the immediate parent.
                
                if levels_up > 0 and len(file_path.parents) >= levels_up:
                    # -1 because parents is 0-indexed (0 is 1 level up)
                    survey_folder = file_path.parents[levels_up - 1]
                    
                    # Double check name consistency (sanity check)
                    if survey_folder.name == top_level_name:
                         resolved_path = survey_folder.resolve()
                         if resolved_path not in seen_paths:
                            seen_paths.add(resolved_path)
                            matching_folders.append(survey_folder)
        except Exception as e:
            logger.error(f"Error processing partition {partition_id}: {e}")

    logger.info(f"Successfully matched {len(matching_folders)} survey directories.")
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


def read_json_questionnaire(survey_path: Path) -> dict:
    """Reads the questionnaire JSON definition."""
    # Try to open the JSON file
    file_path = survey_path / 'Questionnaire' / 'content' / 'document.json'
    try:
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (Exception) as e:
        logger.warning(f"Questionnaire document not found or invalid at {file_path}: {e}")
        return None


def get_questionnaire(data_path: Path, questionnaire_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads and processes a questionnaire from a JSON file located at the specified path.
    Also handles categorization of data.
    """
    q_data = read_json_questionnaire(data_path)

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

        if 'timestamp_utc' in df_para.columns and 'tz_offset' in df_para.columns:
            df_para['timestamp_utc'] = pd.to_datetime(df_para['timestamp_utc'])
            # Only apply if tz_offset is string
            if pd.api.types.is_string_dtype(df_para['tz_offset']):
                 df_para['tz_offset'] = pd.to_timedelta(df_para['tz_offset'].str.replace(':', ' hours ') + ' minutes')
            df_para['timestamp_local'] = df_para['timestamp_utc'] + df_para['tz_offset']

        try:
            parts = parse_filename(data_path.name)
            qnr_name = parts[0]
            qnr_version = parts[1]
            df_para = set_qnr_version(df_para, qnr_name, qnr_version)
        except ValueError:
            logger.warning(f"Could not parse filename '{data_path.name}' for version info")

        if not df_questionnaires.empty:
            q_columns = ['qnr_seq', 'variable_name', "qtype", 'question_type',
                         'answers', 'question_scope',
                         'yes_no_view', 'is_filtered_combobox',
                         'is_integer', 'cascade_from_question_id',
                         'answer_sequence', 'n_answers', 'question_sequence',
                         'qnr', 'qnr_version']
            
            # Ensure columns exist in questionnaire df before selecting
            q_columns = [c for c in q_columns if c in df_questionnaires.columns]

            # Merge
            df_para = df_para.merge(df_questionnaires[q_columns], how='left',
                                    left_on=['param', 'qnr', 'qnr_version'],
                                    right_on=['variable_name', 'qnr', 'qnr_version'])

        # Normalize column names
        df_para.columns = [normalize_column_name(c) for c in df_para.columns]

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
            
            # Handle StataMissingValue objects which are unhashable
            # Replace '.a' with -999999999 and '.' with NaN
            from pandas.io.stata import StataMissingValue

            def replace_stata_missing(val):
                if isinstance(val, StataMissingValue):
                    s_val = str(val)
                    if s_val == '.a':
                        return -999999999
                    elif s_val == '.':
                        return np.nan
                    return np.nan # defaulting other missing values to NaN
                return val

            # Apply only to object columns where StataMissingValue might exist
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].apply(replace_stata_missing)
            
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
    drop_list = ['interview__key', 'sssys_irnd', 'has__errors', 'interview__status', 'assignment__id']

    file_names = get_microdata_file_list(data_path)
    
    # # Pre-calculate masks outside loop
    # # Pre-initialize these variable lists once so they exist when the questionnaire DF is empty
    # # (avoids NameError and avoids recalculating per-file).
    # multi_unlinked_vars = []
    # multi_linked_vars = []
    # list_vars = []
    # gps_vars = []

    # define multi/list question conditions
    if not df_questionnaires.empty:
        # Use boolean indexing
        unlinked_mask = (df_questionnaires["qtype"] == 'MultyOptionsQuestion') & (
            df_questionnaires['is_linked'] == False)
        linked_mask = (df_questionnaires["qtype"] == 'MultyOptionsQuestion') & (
            df_questionnaires['is_linked'] == True)
        list_mask = (df_questionnaires["qtype"] == 'TextListQuestion')
        gps_mask = (df_questionnaires["qtype"] == 'GpsCoordinateQuestion')
        
        # extract multi/list question lists from conditions
        multi_unlinked_vars = df_questionnaires.loc[unlinked_mask, 'variable_name'].tolist()
        multi_linked_vars = df_questionnaires.loc[linked_mask, 'variable_name'].tolist()
        list_vars = df_questionnaires.loc[list_mask, 'variable_name'].tolist()
        gps_vars = df_questionnaires.loc[gps_mask, 'variable_name'].tolist()
    
    # Iterate over each file
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

        # create roster_level from __id columns if on roster level, else '' if main questionnaire file
        roster_ids = [col for col in df.columns if col.endswith("__id") and col != "interview__id"]
        if roster_ids:
            df['roster_level'] = df[roster_ids].apply(lambda row: ",".join(map(str, row)), axis=1)
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
        if isinstance(val, (list, tuple)): return len(val) > 0 # Empty list should be invalid? Legacy: 'return True'
        if isinstance(val, (np.ndarray,)): return val.size > 0
        if isinstance(val, str) and val == '': return False
        # Fallback for other types where equality might be array-like (though unlikely for scalars)
        if hasattr(val, 'size') and hasattr(val, 'shape'): # duck typing for arrays
             return val.size > 0
        
        try:
             if pd.isna(val): return False
        except:
             pass 
        
        # Check for empty string equality safely
        if str(val) == '': return False
        
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
    # Normalize float values that are actually integers (e.g. 1.0 -> 1) before string conversion
    # This ensures "107080102.0" becomes "107080102" matching legacy output
    def normalize_and_stringify(val):
        if isinstance(val, float) and val.is_integer():
             return str(int(val))
        if isinstance(val, (list, tuple, np.ndarray)):
             # If it's a list (from transform_multi), we might need to normalize internal floats tool?
             # Legacy code just did astype(str), which calls str(val).
             # str([1.0, 2.0]) -> "[1.0, 2.0]"
             # str([1, 2]) -> "[1, 2]"
             # So we might need to clean up lists too if we want exact match.
             # However, let's stick to scalar normalization first as that's the primary complaint.
             return str(val)
        return str(val)

    # Use apply for robust conversion
    combined_df['value'] = combined_df['value'].apply(normalize_and_stringify)
    
    return combined_df

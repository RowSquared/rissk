import numpy as np
import json
import pyarrow as pa
import pandas as pd
import zipfile
from io import BytesIO
from loguru import logger
from pathlib import Path
import re
import os
from typing import List, Dict, Optional
from rissk.utils.file_process_utils import (get_file_parts, transform_multi,
                                            set_qnr_version, normalize_column_name,
                                            process_json_structure, get_categories,
                                            update_df_categories)



# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

def get_zip_files(data_dir: Path, survey: str, questionnaires: List[Dict[str, List[int]]]) -> List[Path]:
    """
    Retrieves a list of zip files from the specified directory that match the given pattern.

    Parameters:
    - data_dir (Path): The directory to search for zip files.
    - survey (str): The survey name to match in the file names.
    - questionnaires (List[Dict[str, List[int]]]): A list of dictionaries, each containing a 
      'name' of the questionnaire and a 'VERSION' list to match in the file names.

    Returns:
    - List[Path]: A list of matching zip file paths.
    """
    matching_files = []

    # Iterate through each questionnaire and its associated versions
    for questionnaire in questionnaires:
        name = questionnaire.get('name')
        versions = questionnaire.get('VERSION', [])
        
        # Compile a regex pattern for matching files
        version_pattern = "|".join(map(str, versions))
        pattern = re.compile(rf"{name}_({version_pattern})_.*\.zip")
        
        # List and filter files in the specified directory
        matching_files.extend(
            file_path
            for file_path in data_dir.iterdir()
            if pattern.match(file_path.name)
        )
    
    return matching_files


def extract_zip(file_source_path: Path, file_dest_path: Path):
    """
    Extracts a zip file to the specified destination path.
    If nested zip files are encountered, they are extracted recursively.
    
    Parameters:
    - file_source_path (Path): Path to the source zip file.
    - file_dest_path (Path): Destination directory where files will be extracted.
    """
    password = os.getenv('PASSWORD', None)
    
    try:
        with file_source_path.open(mode='rb') as f:
            zip_data = BytesIO(f.read())
        
        with zipfile.ZipFile(zip_data) as zip_ref:
            for file_info in zip_ref.infolist():
                file_name = file_info.filename
                file_path = file_dest_path / file_name
                
                if file_info.is_dir():
                    file_path.mkdir(parents=True, exist_ok=True)
                else:
                    extracted_data = zip_ref.read(file_name, pwd=password.encode() if password else None)
                    if file_name.endswith('.zip'):
                        nested_dir = file_path.with_suffix('')
                        nested_dir.mkdir(parents=True, exist_ok=True)
                        nested_zip_path = nested_dir / file_path.name
                        with nested_zip_path.open(mode='wb') as nested_f:
                            nested_f.write(extracted_data)
                        extract_zip(nested_zip_path, nested_dir)  # Recursively extract nested zip file
                    else:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        with file_path.open(mode='wb') as extracted_f:
                            extracted_f.write(extracted_data)
        
        logger.info(f'Zip file {file_source_path} extracted successfully to {file_dest_path}')
    except zipfile.BadZipFile:
        logger.error(f'Error: The file {file_source_path} is not a zip file or it is corrupted.')
    except RuntimeError as e:
        logger.error(f'Error: A runtime error occurred - {e}')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {e}')


def get_from_dir(dir_name: str, info: str) -> str:
    """
    Extract information from a directory name formatted as '<QUESTIONAIRE>_<VERSION>_<FORMAT>_<STATUS>'.

    Parameters:
    dir_name (str): The directory name to parse.
    info (str): The type of information to extract ('questionaire', 'version', 'format', 'status').

    Returns:
    str: The extracted information.

    Raises:
    ValueError: If the info parameter is not one of 'questionaire', 'version', 'format', 'status'.
    IndexError: If the directory name does not have the expected format.
    """
    # Map info to the corresponding index
    info_index = {
        'questionaire': 0,
        'version': 1,
        'format': 2,
        'status': 3
    }

    if info not in info_index:
        raise ValueError("info parameter must be one of 'questionaire', 'version', 'format', 'status'")

    # Reverse split to handle potential underscores in QUESTIONAIRE
    parts = dir_name.rsplit('_', 3)
    
    if len(parts) < 4:
        raise IndexError("Directory name does not have the expected format '<QUESTIONAIRE>_<VERSION>_<FORMAT>_<STATUS>'")

    return parts[info_index[info]]

def assign_type(df, dtypes):
    for column in dtypes.index:
        df[column] = df[column].astype(dtypes[column])
    return df


def get_survey_info(survey_files):

    survey_info = {}

    for survey_path in survey_files:
        filename = survey_path.name
        questionnaire, version, file_format, interview_status = get_file_parts(filename)
        qnr_version = f"{questionnaire}_{str(version)}"  

        survey_info[questionnaire] = survey_info.get(questionnaire, {})
        survey_info[questionnaire][qnr_version] = survey_info[questionnaire].get(qnr_version, {})
        survey_info[questionnaire][qnr_version][file_format] = survey_path
    return survey_info



def save_parquet(df, file_path):
    with open(file_path, 'wb') as f:
        if 'answer_sequence' in df.columns:
            df['answer_sequence'] = df['answer_sequence'].apply(str)
        df.to_parquet(f)


def read_microdata_files(s_path, file_name):
    file_path = os.path.join(s_path, file_name)
    if file_name.endswith('.dta'):
        try:
            with open(file_path, 'rb') as f:
                df = pd.read_stata(f, convert_categoricals=False, convert_missing=True)
            # Manage missing values
            df = df.where(df.astype(str) != '.a', -999999999)  # replace '.a' with -999999999 to match tabular export
            df = df.where(df.astype(str) != '.', np.nan)  # replace '.' with np.nan
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    else:
        with open(file_path) as f:
            df = pd.read_csv(f, delimiter='\t')
    return df


def get_microdata_file_list(data_path: Path) -> List[str]:
    """
    Get a list of microdata files in the specified directory, excluding certain files and extensions.

    Parameters:
    data_path (Path): The directory path to search for files.

    Returns:
    List[str]: A list of file names that match the criteria.
    """
    excluded_files = ('interview__', 'assignment__', 'paradata.tab')
    excluded_extensions = ('.dta', '.tab')

    # List comprehension to filter files
    file_names = [
        file.name for file in data_path.iterdir()
        if file.is_file() and file.suffix in excluded_extensions and not any(file.name.startswith(prefix) for prefix in excluded_files)
    ]

    return file_names


def get_microdata(data_path, df_questionnaires):
    drop_list = ['interview__key', 'sssys_irnd', 'has__errors', 'interview__status', 'assignment__id']

    file_names = get_microdata_file_list(data_path)

    # define multi/list question conditions
    if df_questionnaires.empty is False:
        unlinked_mask = (df_questionnaires["qtype"] == 'MultyOptionsQuestion') & (
                df_questionnaires['is_linked'] == False)
        linked_mask = (df_questionnaires["qtype"] == 'MultyOptionsQuestion') & (df_questionnaires['is_linked'] == True)
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

        df = read_microdata_files(data_path, file_name)
        # drop system-generated columns
        df.drop(columns=[col for col in drop_list if col in df.columns], inplace=True)

        # transform multi/list questions
        if df_questionnaires.empty is False:
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
        df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='variable', value_name='value')
        df_long['filename'] = file_name

        all_dfs.append(df_long)
    if len(all_dfs) > 0:

        combined_df = pd.concat(all_dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    # Drop column with null or empty string in value
    # Function to check if the value is not an empty string or NaN
    def is_valid(value):
        if isinstance(value, list):
            return True  # bool(value)  # Not an empty list
        return value != '' and pd.notna(value)  # Not an empty string or NaN

    # Keep rows where the 'value' column passes the is_valid check
    combined_df = combined_df[combined_df['value'].apply(is_valid)]


    questionaire_name = get_from_dir(data_path.name, 'questionaire')
    qnr_version = get_from_dir(data_path.name, 'version')
    combined_df = set_qnr_version(combined_df, questionaire_name, qnr_version)

    # Manage the case questionnaires are not available for the survey
    if df_questionnaires.empty is False:
        roster_columns = [c for c in combined_df.columns if '__id' in c and c != 'interview__id']
        combined_df = combined_df.merge(df_questionnaires, how='left',
                                        left_on=['variable', 'qnr', 'qnr_version'],
                                        right_on=['variable_name', 'qnr', 'qnr_version']).sort_values(
            ['interview__id', 'qnr_seq'] + roster_columns)

    combined_df.reset_index(drop=True, inplace=True)

    # Normalize columns
    combined_df.columns = [normalize_column_name(c) for c in combined_df.columns]

    # Set value column to string for type compatibility
    combined_df['value'] = combined_df['value'].astype(str)
    return combined_df


def get_questionnaire_map(raw_path):
    questionnaire_map = {}
    questionnaire_list = os.listdir(raw_path)
    for questionnaire in questionnaire_list:
        if questionnaire.endswith('.json'):
            file_name = os.path.basename(questionnaire)
            questionnaire_id = file_name.split('_')[0].replace('-', '')
            qnr_version = questionnaire.split('_')[1].replace('.json', '')
            questionnaire_map[questionnaire_id] = {
                'file_name': file_name,
                'qnr_version': qnr_version,
                'file_path': os.path.join(raw_path, file_name)
            }
    return questionnaire_map


def get_questionnaire_id(extracted_path):
    file_path = os.path.join(extracted_path, 'export__info.json')
    with open(file_path, mode='r') as f:
        data = json.load(f)
    return data.get('QuestionnaireId').split("$")[0]


def read_json_questionnaire(survey_path, questionnaire_path=None):
    if questionnaire_path is None:
        file_path = os.path.join(survey_path, 'Questionnaire/content/document.json')
    else:
        questionnaire_id = get_questionnaire_id(survey_path)
        questionnaire_map = get_questionnaire_map(questionnaire_path)
        file_path = questionnaire_map.get(questionnaire_id).get('file_path')
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def read_paradata(survey_path, delimiter='\t'):
    file_path = os.path.join(survey_path, 'paradata.tab')
    with open(file_path, 'r') as f:
        df = pd.read_csv(f, delimiter=delimiter)
    return df

def get_questionnaire(data_path: Path, questionnaire_path: Optional[Path] = None) -> pd.DataFrame:
    """
    This function loads and processes a questionnaire from a JSON file located at the specified path.
    It also handles the categorization of the data.

    Parameters:
    data_path (Path): The path to the directory containing the questionnaire and categories data.
    questionnaire_path (Optional[Path]): The path to the questionnaire JSON file.

    Returns:
    pd.DataFrame: A processed DataFrame containing the questionnaire data.
    """
    q_data = read_json_questionnaire(data_path, questionnaire_path=questionnaire_path)

    qnr_df = pd.DataFrame()

    if q_data is not None:
        question_data = []
        question_counter = 0

        process_json_structure(q_data["Children"], "", question_counter, question_data)

        qnr_df = pd.DataFrame(question_data)
        qnr_df['answer_sequence'] = qnr_df['Answers'].apply(
            lambda x: [int(item['AnswerValue']) for item in x] if x else np.nan)
        qnr_df['n_answers'] = qnr_df['Answers'].apply(lambda x: len(x) if x else np.nan)
        qnr_df['is_linked'] = (qnr_df['LinkedToRosterId'].notna()) | (qnr_df['LinkedToQuestionId'].notna())
        qnr_df['parents'] = qnr_df['parents'].str.lstrip(' > ')
        split_columns = qnr_df['parents'].str.split(' > ', expand=True)
        split_columns.columns = [f"parent_{i + 1}" for i in range(split_columns.shape[1])]
        qnr_df = pd.concat([qnr_df, split_columns], axis=1)
        qmask = qnr_df['QuestionScope'] == 0
        qnr_df['question_sequence'] = qmask.cumsum()
        qnr_df.loc[~qmask, 'question_sequence'] = None
        
    categories_path = data_path / 'Questionnaire' / 'content' / 'Categories'

    if categories_path.exists():
        categories = get_categories(categories_path)
        qnr_df = qnr_df.apply(lambda row: update_df_categories(row, categories), axis=1)

    qnr_df.reset_index(drop=True, inplace=True)
    # Normalize columns
    qnr_df.columns = [normalize_column_name(c) for c in qnr_df.columns]

    questionaire_name = get_from_dir(data_path.name, 'questionaire')
    qnr_version = get_from_dir(data_path.name, 'version')
    qnr_df = set_qnr_version(qnr_df, questionaire_name, qnr_version)
    return qnr_df


def get_paradata(data_path, df_questionnaires):
    """
    This function loads and processes a paradata file from the provided path and merges it with the questionnaire dataframe.
    The function also generates a date-time column from the timestamp and marks whether the answer has changed.

    Parameters:
    para_path (str): A string path to the paradata .csv file.
    df_questionnaires (DataFrame): A Pandas DataFrame containing the questionnaire data.

    Returns:
    df_para (DataFrame): A processed DataFrame containing the merged data from the paradata file and the questionnaire DataFrame.

    """
    df_para = read_paradata(data_path, delimiter='\t')

    # split the parameter column, first from the left, then from the right to avoid potential data entry issues
    df_para[['param', 'answer']] = df_para['parameters'].str.split('\|\|', n=1, expand=True)
    df_para[['answer', 'roster_level']] = df_para['answer'].str.rsplit('||', n=1, expand=True)

    #df_para['roster_level'] = df_para['roster_level'].str.replace("|","")  # if yes/no questions are answered with yes for the first time, "|" will appear in roster

    # generate date-time, TZ not yet considered
    df_para['timestamp_utc'] = pd.to_datetime(df_para['timestamp_utc'])
    df_para['tz_offset'] = pd.to_timedelta(df_para['tz_offset'].str.replace(':', ' hours ') + ' minutes')
    # Adjust the date column by the timezone offset
    df_para['timestamp_local'] = df_para['timestamp_utc'] + df_para['tz_offset']


    questionaire_name = get_from_dir(data_path.name, 'questionaire')
    qnr_version = get_from_dir(data_path.name, 'version')

    df_para = set_qnr_version(df_para, questionaire_name, qnr_version)

    #Merge with questionnaire data
    if df_questionnaires.empty is False:
        q_columns = ['qnr_seq', 'variable_name', "qtype", 'question_type',
                     'answers', 'question_scope',
                     'yes_no_view', 'is_filtered_combobox',
                     'is_integer', 'cascade_from_question_id',
                     'answer_sequence', 'n_answers', 'question_sequence',
                     'qnr', 'qnr_version']
        df_para = df_para.merge(df_questionnaires[q_columns], how='left',
                                left_on=['param', 'qnr', 'qnr_version'],
                                right_on=['variable_name', 'qnr', 'qnr_version'])

    # Normalize column names
    df_para.columns = [normalize_column_name(c) for c in df_para.columns]
    return df_para


def get_dataframes(survey_info):
    """
    Returns dataframes of the paradata, questionnaires, and microdata.

    Parameters:
    save_to_disk: A boolean indicating whether to save the dataframes to disk.
    reload: A boolean indicating whether to reload the data.

    Returns:
    df_paradata, df_questionnaires, df_microdata: Dataframes containing the paradata, questionnaires, and microdata from the different surveys defined in the config.
    """
    dfs_paradata = []
    dfs_questionnaires = []
    dfs_microdata = []
    
    for survey_questionnaire, questionnaires_details in survey_info.items():
        for questionnaires_version, file_paths in questionnaires_details.items():
            tabular_path = file_paths['Tabular']
            paradata_path = file_paths['Paradata']

            try:
                df_questionnaires = get_questionnaire(tabular_path)
            except Exception as e:
                logger.error(f"Failed to load questionnaire for {survey_questionnaire} version {questionnaires_version} from {tabular_path}: {str(e)}")
                raise

            try:
                df_paradata = get_paradata(paradata_path, df_questionnaires)
            except Exception as e:
                logger.error(f"Failed to load paradata for {survey_questionnaire} version {questionnaires_version} from {paradata_path}: {str(e)}")
                raise

            try:
                df_microdata = get_microdata(tabular_path, df_questionnaires)
            except Exception as e:
                logger.error(f"Failed to load microdata for {survey_questionnaire} version {questionnaires_version} from {tabular_path}: {str(e)}")
                raise

            logger.info(f"{survey_questionnaire} with version {questionnaires_version} loaded. "
                        f"\n"
                        f"Paradata shape: {df_paradata.shape} "
                        f"Questionnaires shape: {df_questionnaires.shape} "
                        f"Microdata shape: {df_microdata.shape} ")

            dfs_paradata.append(df_paradata)
            dfs_questionnaires.append(df_questionnaires)
            dfs_microdata.append(df_microdata)

    # create unique dataframe with all surveys
    try:
        dfs_paradata = pd.concat(dfs_paradata)
        dfs_questionnaires = pd.concat(dfs_questionnaires)
        dfs_microdata = pd.concat(dfs_microdata)
    except Exception as e:
        logger.error(f"Failed to concatenate dataframes: {str(e)}")
        raise

    dfs_paradata.reset_index(drop=True, inplace=True)
    dfs_questionnaires.reset_index(drop=True, inplace=True)
    dfs_microdata.reset_index(drop=True, inplace=True)

    return dfs_paradata, dfs_questionnaires, dfs_microdata
    

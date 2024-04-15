import numpy as np
import json
import pyarrow as pa
import pandas as pd
import zipfile
from io import BytesIO
from rissk.utils.file_manager_utils import *
from rissk.utils.file_process_utils import (get_file_parts, transform_multi,
                                            set_survey_name_version, normalize_column_name,
                                            process_json_structure, get_categories,
                                            update_df_categories)


def assign_type(df, dtypes):
    for column in dtypes.index:
        df[column] = df[column].astype(dtypes[column])
    return df


def get_import_path(path, survey_names, **kwargs):
    available_surveys = fs_listdir(path, **kwargs)

    if survey_names == 'all':
        return available_surveys

    import_path = [survey for survey in survey_names if survey in available_surveys]

    if not import_path:
        raise ValueError(f"ERROR: survey path {path} does not exists")

    return import_path


def update_file_dict_version(file_dict, surveys, survey_version, export_path):
    """
    Update the file dictionary based on the surveys specified in the config.
    """

    if surveys != 'all':
        if survey_version is None:
            if len(file_dict[surveys[0]]) > 1:
                raise ValueError(f"There are multiple versions in {export_path}. "
                                 f"Either specify survey_version=all in python main.py i.e. \n"
                                 f"python main.py export_path={export_path} survey_version=all "
                                 f"\n OR provide a path with only one version.")
        elif survey_version == 'all':
            file_dict = {survey: survey_data for survey, survey_data in file_dict.items() if
                         survey in surveys}
        else:
            file_dict = {k: {nk: v for nk, v in nested_dict.items() if nk in survey_version} for
                         k, nested_dict in file_dict.items() if k in surveys}
    return file_dict


def get_file_dict(config):
    """
    Get a dictionary with all zip files from the surveys defined in the config.
    """
    # Get a dictionary with all zip files from the surveys defined in config

    file_dict = {}

    root_path = config['environment']['data']['externals']
    survey_names = config['surveys']
    survey_version = config['survey_version']
    survey_path = config['export_path']
    output_file = config['output_file']

    import_path = get_import_path(root_path, survey_names, **config)

    for survey_name in import_path:
        if os.path.isdir(os.path.join(root_path, survey_name)):
            file_dict[survey_name] = file_dict.get(survey_name, {})

            survey_path = os.path.join(root_path, survey_name)
            for filename in fs_listdir(survey_path, config):
                if filename.endswith('.zip'):

                    try:
                        questionnaire, version, file_format, interview_status = get_file_parts(filename)
                        q_name = f"{questionnaire}_{str(version)}"
                        file_dict[survey_name][q_name] = file_dict[survey_name].get(q_name, {
                            'file_path': survey_path})
                        file_dict[survey_name][q_name][file_format] = filename
                    except ValueError:
                        print(f"WARNING: Survey {survey_name} with version filename {filename} Skipped")
    # Filter out folders without ZIP files.
    file_dict = {k: v for k, v in file_dict.items() if len(v) > 0}

    file_dict = update_file_dict_version(file_dict, survey_names, survey_version, survey_path)
    return file_dict


def load_dataframes(processed_data_path, config):
    file_path = os.path.join(processed_data_path, 'questionnaire.parquet')
    with fs_open(file_path, mode='rb', **config) as f:
        df_questionnaire = pd.read_parquet(f)

    file_path = os.path.join(processed_data_path, 'paradata.parquet')
    with fs_open(file_path, mode='rb', **config) as f:
        df_paradata = pd.read_parquet(f)

    file_path = os.path.join(processed_data_path, 'microdata.parquet')
    with fs_open(file_path, mode='rb', **config) as f:
        df_microdata = pd.read_parquet(f)

    return df_paradata, df_questionnaire, df_microdata


def save_dataframes(df_paradata, df_questionnaires, df_microdata, processed_data_path, config):
    # Create directory if it doesn't exist
    fs_mkrdir(processed_data_path, **config)

    file_path = os.path.join(processed_data_path, 'questionnaire.parquet')
    with fs_open(file_path, **config, mode='wb') as f:
        df_questionnaires.to_parquet(f)

    file_path = os.path.join(processed_data_path, 'paradata.parquet')
    with fs_open(file_path, **config, mode='wb') as f:
        df_paradata.to_parquet(f)

    file_path = os.path.join(processed_data_path, 'microdata.parquet')
    with fs_open(file_path, **config, mode='wb') as f:
        # NOTE! The microdata file is saved with a schema that allows for lists of integers.
        # This is necessary to store the answer_sequence column. You must add this to any other column that is a list.
        schema = pa.schema([pa.field('answer_sequence', pa.list_(pa.int()))])
        df_microdata.to_parquet(f)


def get_data(s_path, s_name, s_version, config):
    """
    This function wraps up the entire process of data extraction from the survey files.
    It calls the get_questionaire, get_paradata, and get_microdata functions in sequence,
    each one with its corresponding arguments.

    Parameters:
    survey_path (str): The directory path where the survey files are located.

    Returns:
    df_paradata (DataFrame): The DataFrame containing all the paradata.
    df_questionnaires (DataFrame): DataFrame containing information about the questionnaire used for the survey.
    df_microdata (DataFrame): The DataFrame containing all the microdata (survey responses).
    """
    df_questionnaires = get_questionaire(s_path, s_name, s_version, **config)
    df_paradata = get_paradata(s_path, df_questionnaires, s_name, s_version, **config)
    df_microdata = get_microdata(s_path, df_questionnaires, s_name, s_version, **config)

    return df_paradata, df_questionnaires, df_microdata


def get_microdata(s_path, df_questionnaires, s_name, s_version, **config):
    drop_list = ['interview__key', 'sssys_irnd', 'has__errors', 'interview__status', 'assignment__id']

    file_names = [file for file in fs_listdir(s_path, **config) if
                  (file.endswith('.dta') or file.endswith('.tab')) and not file.startswith(
                      ('interview__', 'assignment__', 'paradata.tab'))]

    # define multi/list question conditions
    unlinked_mask = (df_questionnaires['type'] == 'MultyOptionsQuestion') & (df_questionnaires['is_linked'] == False)
    linked_mask = (df_questionnaires['type'] == 'MultyOptionsQuestion') & (df_questionnaires['is_linked'] == True)
    list_mask = (df_questionnaires['type'] == 'TextListQuestion')
    gps_mask = (df_questionnaires['type'] == 'GpsCoordinateQuestion')

    # extract multi/list question lists from conditions
    multi_unlinked_vars = df_questionnaires.loc[unlinked_mask, 'variable_name'].tolist()
    multi_linked_vars = df_questionnaires.loc[linked_mask, 'variable_name'].tolist()
    list_vars = df_questionnaires.loc[list_mask, 'variable_name'].tolist()
    gps_vars = df_questionnaires.loc[gps_mask, 'variable_name'].tolist()

    # Iterate over each file
    all_dfs = []
    for file_name in file_names:
        file_path = os.path.join(s_path, file_name)
        if file_name.endswith('.dta'):
            try:
                with fs_open(file_path, mode='rb', **config) as f:
                    df = pd.read_stata(f, convert_categoricals=False, convert_missing=True)
                # Manage missing values
                df = df.where(df.astype(str) != '.a', -999999999)  # replace '.a' with -999999999 to match tabular export
                df = df.where(df.astype(str) != '.', np.nan)  # replace '.' with np.nan
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        else:
            with fs_open(file_path, **config) as f:
                df = pd.read_csv(f, delimiter='\t')

        # drop system-generated columns
        df.drop(columns=[col for col in drop_list if col in df.columns], inplace=True)

        # transform multi/list questions
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

    combined_df = set_survey_name_version(combined_df, s_name, s_version)
    # Manage the case questionnaires are not available for the survey
    if df_questionnaires.empty is False:
        roster_columns = [c for c in combined_df.columns if '__id' in c and c != 'interview__id']
        combined_df = combined_df.merge(df_questionnaires, how='left',
                                        left_on=['variable', 'survey_name', 'survey_version'],
                                        right_on=['variable_name', 'survey_name', 'survey_version']).sort_values(
            ['interview__id', 'qnr_seq'] + roster_columns)

    combined_df.reset_index(drop=True, inplace=True)

    # Normalize columns
    combined_df.columns = [normalize_column_name(c) for c in combined_df.columns]
    return combined_df


def read_json_questionaire(survey_path, config):
    file_path = os.path.join(survey_path, 'Questionnaire/content/document.json')
    with fs_open(file_path, config, mode='r') as f:
        data = json.load(f)
    return data


def read_paradata(survey_path, delimiter='\t', **config):
    file_path = os.path.join(survey_path, 'paradata.tab')
    with fs_open(file_path, config, mode='r') as f:
        df = pd.read_csv(f, delimiter=delimiter)
    return df


def get_questionaire(s_path, s_name, s_version, **config):
    """
    This function loads and processes a questionnaire from a JSON file located at the specified path.
    It also handles the categorization of the data.

    Parameters:
    survey_path (str): The path to the directory containing the questionnaire and categories data.

    Returns:
    qnr_df (DataFrame): A processed DataFrame containing the questionnaire data.

    """

    q_data = read_json_questionaire(s_path, config)
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
    categories_path = os.path.join(s_path, 'Questionnaire/content/Categories')

    if fs_exists(categories_path):
        categories = get_categories(categories_path)
        qnr_df = qnr_df.apply(lambda row: update_df_categories(row, categories), axis=1)

    qnr_df.reset_index(drop=True, inplace=True)
    # Normalize columns
    qnr_df.columns = [normalize_column_name(c) for c in qnr_df.columns]
    qnr_df = set_survey_name_version(qnr_df, s_name, s_version)
    return qnr_df


def get_paradata(s_path, df_questionnaires, s_name, s_version, **kwargs):
    """
    This function loads and processes a paradata file from the provided path and merges it with the questionnaire dataframe.
    The function also generates a date-time column from the timestamp and marks whether the answer has changed.

    Parameters:
    para_path (str): A string path to the paradata .csv file.
    df_questionnaires (DataFrame): A Pandas DataFrame containing the questionnaire data.

    Returns:
    df_para (DataFrame): A processed DataFrame containing the merged data from the paradata file and the questionnaire DataFrame.

    """
    df_para = read_paradata(s_path, delimiter='\t')

    # split the parameter column, first from the left, then from the right to avoid potential data entry issues
    df_para[['param', 'answer']] = df_para['parameters'].str.split('\|\|', n=1, expand=True)
    df_para[['answer', 'roster_level']] = df_para['answer'].str.rsplit('||', n=1, expand=True)

    #df_para['roster_level'] = df_para['roster_level'].str.replace("|","")  # if yes/no questions are answered with yes for the first time, "|" will appear in roster

    # generate date-time, TZ not yet considered
    df_para['timestamp_utc'] = pd.to_datetime(df_para['timestamp_utc'])
    df_para['tz_offset'] = pd.to_timedelta(df_para['tz_offset'].str.replace(':', ' hours ') + ' minutes')
    # Adjust the date column by the timezone offset
    df_para['timestamp_local'] = df_para['timestamp_utc'] + df_para['tz_offset']

    df_para = set_survey_name_version(df_para, s_name, s_version)

    #Merge with questionnaire data
    if df_questionnaires.empty is False:
        q_columns = ['qnr_seq', 'variable_name', 'type', 'question_type',
                     'answers', 'question_scope',
                     'yes_no_view', 'is_filtered_combobox',
                     'is_integer', 'cascade_from_question_id',
                     'answer_sequence', 'n_answers', 'question_sequence',
                     'survey_name', 'survey_version']
        df_para = df_para.merge(df_questionnaires[q_columns], how='left',
                                left_on=['param', 'survey_name', 'survey_version'],
                                right_on=['variable_name', 'survey_name', 'survey_version'])

    # Normalize column names
    df_para.columns = [normalize_column_name(c) for c in df_para.columns]
    return df_para


def get_dataframes(file_dict, source_path, dest_path, config, save_to_disk=True, reload=False):
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
    for survey_name, survey in file_dict.items():

        for survey_version, files in survey.items():
            survey_source_path = os.path.join(source_path, survey_name, survey_version)
            survey_dest_path = os.path.join(dest_path, survey_name, survey_version)
            print(f"Improting from {survey_source_path} to {survey_dest_path})")
            if reload is False and os.path.isdir(survey_dest_path):
                df_paradata, df_questionnaires, df_microdata = load_dataframes(survey_dest_path, config)
            else:
                df_paradata, df_questionnaires, df_microdata = get_data(survey_source_path, survey_name, survey_version,
                                                                        config)
                if save_to_disk:
                    save_dataframes(df_paradata, df_questionnaires, df_microdata, survey_dest_path, config)

            print(f"{survey_name} with version {survey_version} loaded. "
                  f"\n"
                  f"Paradata shape: {df_paradata.shape} "
                  f"Questionnaires shape: {df_questionnaires.shape} "
                  f"Microdata shape: {df_microdata.shape} "
                  )

            dfs_paradata.append(df_paradata)
            dfs_questionnaires.append(df_questionnaires)
            dfs_microdata.append(df_microdata)

    # create unique dataframe with all surveys
    dfs_paradata = pd.concat(dfs_paradata)
    dfs_questionnaires = pd.concat(dfs_questionnaires)
    dfs_microdata = pd.concat(dfs_microdata)

    dfs_paradata.reset_index(drop=True, inplace=True)
    dfs_questionnaires.reset_index(drop=True, inplace=True)
    dfs_microdata.reset_index(drop=True, inplace=True)

    return dfs_paradata, dfs_questionnaires, dfs_microdata


def extract_zip(file_source_path, file_dest_path, **config):
    password = config.get('password', None)
    try:
        with fs_open(file_source_path, mode='rb', **config) as f:
            zip_data = BytesIO(f.read())

        with zipfile.ZipFile(zip_data) as zip_ref:
            for file_info in zip_ref.infolist():
                file_name = file_info.filename
                extracted_data = zip_ref.read(file_name, pwd=password.encode() if password else None)
                file_path = os.path.join(file_dest_path, file_name)

                if os.path.basename(file_name) != file_name and file_name.endswith('.zip') is False:
                    dir_path = os.path.dirname(file_path)
                    fs_mkrdir(dir_path, **config)
                elif file_name.endswith('.zip'):
                    # Create a new directory for the nested zip file
                    nested_dir = os.path.splitext(file_path)[0]
                    fs_mkrdir(nested_dir, **config)

                    # Save the nested zip file
                    with fs_open(file_path, mode='wb', **config) as f:
                        f.write(extracted_data)

                    # Recursively extract the nested zip file
                    extract_zip(file_path, nested_dir, **config)
                elif file_info.is_dir():
                    # If it's a directory, recursively call extract_zip on each file in the directory
                    for nested_file in fs_listdir(file_path, **config):
                        nested_file_path = os.path.join(file_path, nested_file)
                        extract_zip(nested_file_path, file_dest_path, **config)
                else:
                    with fs_open(file_path, mode='wb', **config) as f:
                        f.write(extracted_data)

        print(f'Zip file {file_source_path} extracted and extracted files uploaded successfully to {file_dest_path}')
    except Exception as e:
        print(f'Error: {e}')


def extract_survey(file_dict, file_dest_path, **config):
    """
    Extracts the contents of the zip files to a target directory.

    Parameters:
    overwrite_dir: A boolean indicating whether to overwrite the existing directory.
    """
    extract = config.get('extract', True)
    overwrite_dir = config.get('overwrite_dir', False)

    if extract:
        fs_mkrdir(file_dest_path, **config)
        for survey_name, survey in file_dict.items():
            target_dir = os.path.join(file_dest_path, survey_name)

            if overwrite_dir and fs_exists(target_dir):
                pass  # shutil.rmtree(target_dir)

            # Create a new target directory if it does not yet exist
            fs_mkrdir(target_dir, **config)

            for survey_version, files in survey.items():
                file_path = files['file_path']
                dest_path = os.path.join(target_dir, survey_version)
                source_path1 = os.path.join(file_path, files['Paradata'])
                source_path2 = os.path.join(file_path, files['Tabular'])
                if overwrite_dir and fs_exists(dest_path):
                    pass  # shutil.rmtree(dest_path)

                fs_mkrdir(dest_path, **config)
                extract_zip(source_path1, dest_path, **config)
                extract_zip(source_path2, dest_path, **config)
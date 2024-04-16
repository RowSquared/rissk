import os
import pandas as pd
from rissk.utils.file_manager_utils import fs_listdir, fs_open


def set_survey_name_version(df, survey_name, survey_version):
    df['survey_name'] = survey_name
    df['survey_version'] = survey_version
    return df


def normalize_column_name(s):
    """
    This function converts any string with capital letters to a string all lowercase with a "_" before any previously capital letter.

    Parameters:
    s (str): The string to convert.

    Returns:
    new_s (str): The converted string.
    """
    new_s = ""
    for i, char in enumerate(s):
        if char.isupper():
            # Add underscore only if it's not the first or last character
            if i != 0 and i != len(s) - 1:
                new_s += "_"
            new_s += char.lower()
        else:
            new_s += char
    return new_s


def transform_multi(df, variable_list, transformation_type):
    """
    This function takes a DataFrame and a list of variable names and applies a transformation depending on
    transformation_type to the variables in the DataFrame that start with the given variable names.

    The transformation can be either 'unlinked,' 'linked,' 'list,' or 'gps.'

    Parameters:
    df (DataFrame): The DataFrame to be transformed.
    variable_list (list): The list of variable names to be transformed.
    transformation_type (str): The type of transformation to apply. Must be 'unlinked,' 'linked,' 'list,' or 'gps.'

    Returns:
    DataFrame: The transformed DataFrame.

    Raises:
    ValueError: If transformation_type is not 'unlinked,' 'linked,' 'list,' or 'gps.'
    """
    if transformation_type not in ['unlinked', 'linked', 'list', 'gps']:
        raise ValueError("transformation_type must be either 'unlinked', 'linked', 'list', or 'gps'")

    transformed_df = pd.DataFrame(index=df.index)  # DataFrame for storing transformations

    for var in variable_list:
        if var in df.columns:
            # Drop the target column, should it exist (only text list question on a linked roster)
            df = df.drop(var, axis=1)

        related_cols = [col for col in df.columns if col.startswith(f"{var}__")]

        if related_cols:
            transformation = [[] for _ in range(len(df))] \
                if transformation_type != 'gps' \
                else ['' for _ in range(len(df))]

            for col in related_cols:

                if transformation_type == 'unlinked':
                    suffix = int(col.split('__')[1].replace('n', '-'))
                    mask = df[col] > 0
                    transformation = [x + [suffix] if mask.iloc[i] else x for i, x in enumerate(transformation)]
                elif transformation_type == 'linked':
                    # !NOTE! if you add the (df[col] != -999999999) filter it removes also list that not only
                    # contains -999...
                    mask = (df[col].notna())  # & (df[col] != -999999999)
                    transformation = [x + [df.at[i, col]] if mask.iloc[i] else x for i, x in enumerate(transformation)]
                elif transformation_type == 'list':
                    mask = (df[col] != '##N/A##') & (df[col] != '')
                    transformation = [x + [df.at[i, col]] if mask.iloc[i] else x for i, x in enumerate(transformation)]
                elif transformation_type == 'gps':
                    transformation = [x + (',' if x else '') + (str(df.at[i, col])
                                                                if pd.notna(df.at[i, col])
                                                                   and df.at[i, col] not in ['##N/A##', -999999999]
                                                                else '') for i, x in enumerate(transformation)]

            def remove_unset_value(sub_list):
                sub = list(filter(lambda v: v not in [-999999999, '##N/A##'], sub_list))
                sub = [ele if ele != [] else '##N/A##' for ele in sub]
                sub = sub if sub != [] and list(set(sub)) != ['##N/A##'] else '##N/A##'
                return sub

            transformation = [remove_unset_value(x)
                              if x else float('nan') for x in transformation] if transformation_type != 'gps' else [
                x if x else '' for x in transformation]
            transformed_df[var] = transformation  # Add the transformation to the transformed DataFrame
            df = df.drop(related_cols, axis=1)  # Drop the original columns

    df = pd.concat([df, transformed_df], axis=1)  # Concatenate the original DataFrame with the transformations

    return df.copy()


def process_json_structure(children, parent_group_title, counter, question_data):
    """
    This function processes the JSON structure of a questionnaire, collecting information about the questions.

    Parameters:
    children (list): The children nodes in the current JSON structure.
    parent_group_title (str): The title of the parent group for the current child nodes.
    counter (int): A counter to keep track of the sequence of questions.
    question_data (list): A list where data about each question is appended as a dictionary.

    Returns:
    counter (int): The updated counter value after processing all children nodes.

    """
    for child in children:
        if "$type" in child:
            question_data.append({
                "qnr_seq": counter,
                "VariableName": child.get("VariableName"),
                "qtype": child["$type"],
                "QuestionType": child.get("QuestionType"),
                "Answers": child.get("Answers"),
                "Children": child.get("Children"),
                "ConditionExpression": child.get("ConditionExpression"),
                "HideIfDisabled": child.get("HideIfDisabled"),
                "Featured": child.get("Featured"),
                "Instructions": child.get("Instructions"),
                "Properties": child.get("Properties"),
                "PublicKey": child.get("PublicKey"),
                "QuestionScope": child.get("QuestionScope"),
                "QuestionText": child.get("QuestionText"),
                "StataExportCaption": child.get("StataExportCaption"),
                "VariableLabel": child.get("VariableLabel"),
                "IsTimestamp": child.get("IsTimestamp"),
                "ValidationConditions": child.get("ValidationConditions"),
                "YesNoView": child.get("YesNoView"),
                "IsFilteredCombobox": child.get("IsFilteredCombobox"),
                "IsInteger": child.get("IsInteger"),
                "CategoriesId": child.get("CategoriesId"),
                "Title": child.get("Title"),
                "IsRoster": child.get("IsRoster"),
                "LinkedToRosterId": child.get("LinkedToRosterId"),
                "LinkedToQuestionId": child.get("LinkedToQuestionId"),
                "CascadeFromQuestionId": child.get("CascadeFromQuestionId"),
                "parents": parent_group_title
            })
            counter += 1

        if "Children" in child:
            child_group_title = child.get("Title", "")
            counter = process_json_structure(child["Children"], parent_group_title + " > " + child_group_title, counter,
                                             question_data)

    return counter


def get_categories(directory, **config):
    """
    This function retrieves categories from Excel files within a directory.

    Parameters:
    directory (str): The directory where the category Excel files are stored.

    Returns:
    dict: A dictionary containing category data. Each key represents a filename, and each value is another dictionary
    containing 'n_answers' and 'answer_sequence' which represents the number of answers and the sequence of the answer IDs
    respectively.

    """
    categories = {}

    files = [f for f in fs_listdir(directory, **config) if f.endswith('.xlsx') or f.endswith('.xls')]
    for file in files:
        file_path = os.path.join(directory, file)
        with fs_open(file_path, **config, mode='r') as f:
            df = pd.read_excel(f)
        n_answers = df.shape[0]
        answer_sequence = df['id'].tolist()
        categories[file] = {'n_answers': n_answers, 'answer_sequence': answer_sequence}
    return categories


def update_df_categories(row, categories):
    """
    This function updates a DataFrame row with category information if applicable.

    Parameters:
    row (Series): The Questioner DataFrame row to be updated.
    categories (dict): A dictionary containing category data, keys are 'CategoriesId'.

    Returns:
    Series: The updated DataFrame row.

    """
    if row['CategoriesId'] in categories:
        row['n_answers'] = categories[row['CategoriesId']]['n_answers']
        row['answer_sequence'] = categories[row['CategoriesId']]['answer_sequence']
    return row


def get_file_parts(filename):
    # Remove ".zip" and split by "_"
    filename_parts = filename[:-4].split("_")
    if len(filename_parts) < 4:
        raise ValueError(f"ERROR: {filename} Not a valid Survey Solutions export file.")

    version, file_format, interview_status = filename_parts[-3:]
    try:
        version = int(version)
    except ValueError:
        raise ValueError(f"ERROR: {filename} Not a valid Survey Solutions export file. Version not found.")
    questionnaire = "_".join(filename_parts[:-3])
    # Test input file has the correct name
    if file_format not in ["Tabular", "STATA", "SPSS", "Paradata"]:
        raise ValueError(f"ERROR: {filename} Not a valid Survey Solutions export file. Export type not found")

    if interview_status not in ["Approved", "InterviewerAssigned", "ApprovedBySupervisor", "ApprovedByHQ", "All",
                                'ApprovedByHeadquarters']:
        raise ValueError(f"ERROR: {filename} Not a valid Survey Solutions export file. Interview status not found.")

    file_format = file_format if file_format == 'Paradata' else 'Tabular'
    return questionnaire, version, file_format, interview_status



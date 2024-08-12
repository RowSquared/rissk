from pathlib import Path
import yaml
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

env_file_path = PROJ_ROOT / 'env.yaml'
def parse_questionaire(env_var) -> list:
    # Ensure QUESTIONAIRE is always a list of dictionaries
    questionaire = env_var.get('QUESTIONAIRE', [])
    if isinstance(questionaire, list):
        parsed_questionaire = []
        for item in questionaire:
            if isinstance(item, dict):
                name = item.get('name')
                version = item.get('VERSION', [])
                if isinstance(version, int):
                    version = [version]
                parsed_questionaire.append({
                    'name': name,
                    'VERSION': version
                })
    return parsed_questionaire

with open(env_file_path, 'r') as file:
    env_data = yaml.safe_load(file)

# Load variables
SURVEY = env_data.get('SURVEY')  # This will be a string
QUESTIONAIRE = parse_questionaire(env_data)  # Parse the new structure

# Example usage of parsed data
for item in QUESTIONAIRE:
    name = item['name']
    versions = item['VERSION']
    print(f"Questionnaire: {name}, Versions: {versions}")




DATA_DIR = PROJ_ROOT / 'data' / SURVEY
EXTERNAL_DATA_DIR = DATA_DIR / "00_EXTERNAL"
RAW_DATA_DIR = DATA_DIR / "10_RAW"
INTERIM_DATA_DIR = DATA_DIR / "20_INTERIM"
PROCESSED_DATA_DIR = DATA_DIR / "30_PROCESSED"


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

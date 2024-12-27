from rissk.config import SURVEY, DATA_DIR
from ploomber.clients import LocalStorageClient, GCloudStorageClient, S3Client


def get_local():
    """Returns local client
    """
    return LocalStorageClient(DATA_DIR, path_to_project_root=DATA_DIR)


def get_s3():
    """Returns S3 client
    """
    # assumes your environment is already configured, you may also pass the
    # json_credentials_path
    return S3Client(bucket_name='surveytool', parent=f'{SURVEY}/latest')


def get_gcloud():
    """Returns google cloud storage client
    """
    # assumes your environment is already configured, you may also pass the
    # json_credentials_path
    return GCloudStorageClient(bucket_name='surveytool',
                               parent=f'{SURVEY}/latest')
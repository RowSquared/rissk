import os
import s3fs


def fs_exists(path, key=None, secret=None, is_local=True, **kwargs):
    """
    Check if a file or directory exists at the specified path.

    This function supports both local and S3 file systems. For S3 like file system, it requires credentials (key and secret).

    Parameters:
    path (str): The path of the file or directory to check.
    key (str, optional): The AWS access key ID. Default is None.
    secret (str, optional): The AWS secret access key. Default is None.
    is_local (bool, optional): A flag indicating if the file system is local or S3. Default is True.
    **kwargs: Arbitrary keyword arguments for s3fs.S3FileSystem.

    Returns:
    bool: True if the file or directory exists, False otherwise.
    """
    if is_local is True:
        return os.path.exists(path)
    else:
        fs = s3fs.S3FileSystem(anon=False, key=key, secret=secret)
        return fs.exists(path)


def fs_mkrdir(path, key=None, secret=None, is_local=True, **kwargs):
    if is_local is True:
        os.makedirs(path, exist_ok=True)
    else:
        fs = s3fs.S3FileSystem(anon=False, key=key, secret=secret)
        fs.mkdir(path, exist_ok=True)


def fs_listdir(path, key=None, secret=None, is_local=True, **kwargs):
    if is_local is True:
        return os.listdir(path)
    else:
        fs = s3fs.S3FileSystem(anon=False, key=key, secret=secret)
        return fs.ls(path)


def fs_open(file_path, key=None, secret=None, mode='r', is_local=True, **kwargs):
    if is_local is True:
        return open(file_path, mode)
    else:
        fs = s3fs.S3FileSystem(anon=False, key=key, secret=secret)
        return fs.open(file_path, mode)



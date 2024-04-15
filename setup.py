from setuptools import find_packages, setup

# Read requirements from requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='rissk',
    version='0.1.2',
    description='Automatically identify at-risk interviews from your Survey Solutions export files.',
    author='rowsquared',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    packages=find_packages(where='rissk'),
    package_dir={'': 'rissk'},
    install_requires=requirements,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)

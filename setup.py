from setuptools import find_packages, setup
from typing import List

# It's good practice to use uppercase for constants
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return a list of requirements.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Strip whitespace and remove empty lines
        requirements = [req.strip() for req in requirements if req.strip()]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Padma',
    author_email='dhakappa.padma@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)



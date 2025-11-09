from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e'

def get_requirements(file_path : str) ->List[str]:
    '''
    This function returns the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name = 'Car Price Project',
    version = '0.0.1',
    author= 'Blessington',
    author_email= 'blessington1010@gmail.com',
    install_requires = get_requirements('requirements.txt')
)

#THIS BELOW WORKS - WHEN RUNNING THIS FILE ONLY -> python setup.py
#THE ABOVE WORKS - WHEN THE FILE IS IMPORTED IN ANOTHER FILE
# if __name__=='__main__':
#     setup(
#     name = 'Car Price Project',
#     version = '0.0.1',
#     author= 'Blessington',
#     author_email= 'blessington1010@gmail.com',
#     install_requires = get_requirements('requirements.txt')
# )
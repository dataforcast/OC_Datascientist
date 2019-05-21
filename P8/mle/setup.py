

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='adanet',
    version='0.1',
    author = 'F. Bangui',
    author_email = 'fbt_telecom@yahoo.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Adanet in Cloud ML',
    requires=[]
)

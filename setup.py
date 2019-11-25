from distutils.core import setup
from setuptools import find_packages


setup(
    name='detection_metrics',
    version='1',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Alex Korovko',
    author_email='korovkoalexander@yandex.ru',
    description='',
    install_requires=[
        "numpy>=1.15.4",
]
)

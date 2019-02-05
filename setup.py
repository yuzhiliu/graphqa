#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
        'Click>=6.0',
        'bumpversion==0.5.3',
        'wheel==0.32.1',
        'watchdog==0.9.0',
        'flake8==3.5.0',
        'tox==3.5.2',
        'coverage==4.5.1',
        'sphinx==1.8.1',
        'twine==1.12.1',
        'pytest==3.8.2',
        'pytest-runner==4.2',
        'pyyaml>=4.2b1',
        'tensorflow==1.12.0',
        'tqdm==4.29.1',
        'neo4j-driver==1.6.2',
        'fuzzywuzzy==0.17.0',
        'bert-serving-server==1.7.9',
        'bert-serving-client==1.7.9'
        ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Yuzhi Liu",
    author_email='liuyuzhi83@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD-2-Clause',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    description="A question answering (QA) system built ove knowledge graphs (KG).",
    entry_points={
        'console_scripts': [
            'graphqa=graphqa.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='graphqa',
    name='graphqa',
    packages=find_packages(include=['graphqa']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/yuzhiliu/graphqa',
    version='0.1.0',
    zip_safe=False,
)

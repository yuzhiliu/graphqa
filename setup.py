from setuptools import setup, find_packages

requires = [
    'numpy',
    'argparse',
    'nose'
]

def readme():
    with open('README.rst') as f:
        return f.read()

#if sys.version_info < (3, 2):
#    requires.append('futures==2.2')

setup(name='graphqa',
    version='0.1',
    description='graphqa is a question answering (QA) system built ove
    knowledge graphs (KG).',
    url='https://github.com/sdliuyuzhi/graphqa',
    author='Yuzhi Liu',
    author_email='liuyuzhi83@gmail.com',
    license='BSD 2-Clause License',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    test_suite='nose.collector',
    install_requires=requires,
    entry_points={
        'console_scripts': ['graphqa=graphqa.app:main'],
    },
    zip_safe=False,
    keywords=['knowledge graph', 'question', 'answer', 'QA', 'NLP']
)


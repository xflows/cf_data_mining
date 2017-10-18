from setuptools import setup, find_packages
import os

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
]

dist = setup(
    name='cf_data_mining',
    version='0.1',
    license='MIT License',
    description='Package providing basic data mining widgets (based on scikit-learn) for ClowdFlows 2.0',
    url='https://github.com/xflows/cf_data_mining',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'cf_core',
        'scipy',
        'numpy',
        'scikit-learn==0.15.2',
        'pydot==1.0.2'
    ]
)

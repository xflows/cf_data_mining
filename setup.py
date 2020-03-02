from setuptools import setup, find_packages

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
]

dist = setup(
    name='cf_data_mining',
    version='0.1.3',
    license='MIT License',
    description='Package providing basic data mining widgets (based on scikit-learn) for ClowdFlows >= 2',
    url='https://github.com/xflows/cf_data_mining',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        # 'cf_core',
        'scipy',
        'numpy',
        'scikit-learn',
        'pydot'
    ]
)

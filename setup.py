from setuptools import setup

setup(
    name='cf_data_mining',
    version='0.1',
    include_package_data=True,
    license='MIT License',
    description='Package providing basic data mining widgets (based on scikit-learn) for ClowdFlows 2.0',
    
    install_requires=[
          'scikit-learn>=0.14.1',
    ],
    author='',
    author_email='',
)

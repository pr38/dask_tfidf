from setuptools import setup


install_requires = [
    "dask[array]>=2.4.0",
    "dask_ml",
    "sparse",
    "numpy",
    "scikit-learn"
]

setup(
    name="dask_tfidf",
    version='0.0.1', 
    description="A Dask native implementation of 'Term Frequency Inverse Document Frequency' for dask-ml and scikit-learn",
    url="https://github.com/pr38/dask_tfidf",
    author="Pouya Rezvanipour",
    author_email="pouyar3@yahoo.com",
    license="BSD",
        classifiers=[
        "Programming Language :: Python :: 3",
        'Development Status :: 3 - Alpha',
    ],
    install_requires = install_requires,
    packages=["dask_tfidf"],

)
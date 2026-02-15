from setuptools import setup, find_packages

setup(
    name="bbo",
    version="0.1.0",
    description="Query complexity for classification via MDS embeddings of black-box generative models",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "pandas>=2.0",
        "tqdm>=4.65",
        "joblib>=1.2",
    ],
)

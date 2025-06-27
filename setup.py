from setuptools import setup, find_packages

setup(
    name="promptlearn",
    version="0.1.0",
    description="Prompt-based estimators like PromptClassifier and PromptRegressor for LLM-driven ML pipelines.",
    author="Fredrik Linaker",
    author_email="fredrik.linaker@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "openai",
        "pandas"
    ],
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Framework :: Scikit-Learn",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)

from setuptools import setup, find_packages

# Load version
version_ns = {}
with open("promptlearn/version.py") as f:
    exec(f.read(), version_ns)

setup(
    name="promptlearn",
    version=version_ns["__version__"],
    description="LLM-powered estimators for scikit-learn pipelines",
    long_description=open("README.md", 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    author="Fredrik Linaker",
    author_email="fredrik.linaker@gmail.com",
    url="https://github.com/frlinaker/promptlearn",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "openai",
        "pandas",
        "numpy",
        "joblib",
        "tqdm"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)

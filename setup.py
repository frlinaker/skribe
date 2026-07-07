from setuptools import setup, find_packages

# Load version
version_ns = {}
with open("skribe/version.py") as f:
    exec(f.read(), version_ns)

setup(
    name="skribe",
    version=version_ns["__version__"],
    description="LLM-powered estimators that inscribe prediction logic as standalone Python functions",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Fredrik Linaker",
    author_email="fredrik.linaker@gmail.com",
    url="https://github.com/frlinaker/skribe",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    install_requires=["scikit-learn", "litellm", "pandas", "numpy", "joblib"],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "skribe=skribe.cli:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

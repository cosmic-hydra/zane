"""
Setup script for AI Drug Discovery Platform
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="zane",
    version="1.0.0",
    author="AI Drug Discovery Team",
    description="State-of-the-art AI-powered drug discovery platform with self-learning capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cosmic-hydra/zane",
    packages=find_packages(),
    package_data={"drug_discovery.native": ["*.cpp"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "integrations": [
            "aizynthfinder>=4.3.0",
            "gt4sd>=1.0.0",
            "guacamol>=0.5.5",
            "moses>=0.10.0",
            "transformers>=4.30.0",
        ],
        "dashboard": [
            "rich>=13.7.0",
            "requests>=2.31.0",
            "beautifulsoup4>=4.12.0",
            "pypdf>=4.2.0",
            "python-dotenv>=1.0.0",
            "cerebras-cloud-sdk>=1.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "zane=drug_discovery.cli:main",
            "drug-discovery=drug_discovery.cli:main",
        ],
    },
)

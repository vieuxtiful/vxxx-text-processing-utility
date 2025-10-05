"""
Setup script for Text Processing Utilities
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="text-processing-utils",
    version="1.0.0",
    author="LexiQ Team",
    author_email="contact@lexiq.dev",
    description="A comprehensive collection of text processing utilities for NLP and text analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lexiq-team/text-processing-utils",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=[
        # No external dependencies - pure Python implementation
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-benchmark>=3.4",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "text-stats=text_processing_utils:cli_text_stats",
            "text-clean=text_processing_utils:cli_text_clean",
        ],
    },
    keywords="text processing nlp tokenization cleaning normalization similarity keywords",
    project_urls={
        "Bug Reports": "https://github.com/lexiq-team/text-processing-utils/issues",
        "Source": "https://github.com/lexiq-team/text-processing-utils",
        "Documentation": "https://text-processing-utils.readthedocs.io/",
    },
)

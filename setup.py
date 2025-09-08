# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hmpi-calculator",
    version="1.0.0",
    author="Environmental Data Analytics Team",
    author_email="team@environmental-analytics.com",
    description="Heavy Metal Pollution Indices Calculator for Groundwater Assessment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/hmpi-calculator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Environmental Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "folium>=0.14.0",
        "streamlit-folium>=0.13.0",
        "scikit-learn>=1.3.0",
        "openpyxl>=3.1.0",
        "xlrd>=2.0.1",
        "Pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jinja2>=3.1.0",
        "pyyaml>=6.0",
        "python-dateutil>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hmpi-app=app:main",
        ],
    },
)


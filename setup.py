"""Setup script for NetworkEcon."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="networkecon",
    version="0.1.0",
    author="NetworkEcon Contributors",
    description="Network Economics Toolkit — graph theory, centrality, diffusion, peer effects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/networkecon/networkecon",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.4",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
    },
)

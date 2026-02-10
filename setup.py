"""
Setup configuration for Advanced Threat Detection System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advanced-threat-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI/ML-powered network threat detection system for real-time analysis",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/advanced-threat-detection",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/advanced-threat-detection/issues",
        "Documentation": "https://github.com/yourusername/advanced-threat-detection#readme",
        "Source Code": "https://github.com/yourusername/advanced-threat-detection",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="threat-detection machine-learning deep-learning cybersecurity network-security anomaly-detection intrusion-detection",
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "full": read_requirements("requirements-full.txt"),
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    py_modules=[
        "threat_detection_system",
        "deep_learning_detector",
        "production_integration",
    ],
    entry_points={
        "console_scripts": [
            "threat-detect=threat_detection_system:main",
            "threat-monitor=production_integration:demonstrate_production_pipeline",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

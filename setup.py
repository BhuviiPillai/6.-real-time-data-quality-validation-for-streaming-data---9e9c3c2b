from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="realtime_data_quality",
    version="0.1.0",
    author="AI Data Validator",
    author_email="example@example.com",
    description="Real-Time Data Quality Validation for Streaming Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/realtime_data_quality",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
) 
from setuptools import setup, find_packages

setup(
    name="deeplib",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "numpy>=1.26.4",
    ],
    author="Dmitry Aspsiov",
    author_email="dmitry.aspisov@gmail.com",
    description="A deep learning library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aspisov/deeplib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
from setuptools import setup, find_packages

setup(
    name="playground",
    version="0.1",
    description="My personal playground.",
    author="Owkin, Ridouane GHERMI",
    url="https://github.com/rghermi-owkin/playground",
    python_requires=">=3.7",
    packages=find_packages(),
    long_description=open("README.md").read(),
)

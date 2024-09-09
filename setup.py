import setuptools
import shutil
import os

from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel

with open("README.md", "r") as fh:
    long_description = fh.read()


def clean_folders():
    print("Removing temporary folders.")

    TMP_FOLDERS = ["prompt_caller.egg-info", "build"]
    for folder in TMP_FOLDERS:
        if os.path.exists(folder):
            shutil.rmtree(folder)


class SdistCommand(sdist):
    """Custom build command."""

    def run(self):
        sdist.run(self)


class BdistWheelCommand(bdist_wheel):
    """Custom build command."""

    def run(self):
        bdist_wheel.run(self)
        clean_folders()


setuptools.setup(
    name="prompt_caller",
    version="0.0.2",
    author="Thiago Nepomuceno",
    author_email="thiago@neps.academy",
    description="This package is responsible for calling prompts in a specific format. It uses LangChain and OpenAI API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/ThiNepo/prompt-caller",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pyyaml>=6.0.2",
        "python-dotenv>=1.0.1",
        "Jinja2>=3.1.4",
        "langchain-openai>=0.1.23",
    ],
    cmdclass={"sdist": SdistCommand, "bdist_wheel": BdistWheelCommand},
)

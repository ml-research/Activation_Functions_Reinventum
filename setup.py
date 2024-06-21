from pathlib import Path
from setuptools import setup, find_packages
from distutils.command.clean import clean



degrees = [(5, 4), (7, 6)]
name='activation-functions'

long_description = ""

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.readlines()


class clean_all(clean):
    def run(self):
        self.all = True
        super().run()
        import shutil
        import os
        egginf = name.replace('-', '_')
        shutil.rmtree(egginf + '.egg-info')
        shutil.rmtree('dist')
        print("Cleaned everything")

setup(
    name=name,
    version=1,
    author="Quentin Delfosse, Patrick Schramowski",
    author_email="quentin.delfosse@cs.tu-darmstadt.de",
    description="Activations functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/k4ntz/activation_functions",
    packages=find_packages(exclude=["tests"]),
    package_data={'': ['*.json']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License"
    ],
    install_requires=requirements,
    ext_modules= [],
    cmdclass={
        'clean': clean_all
    },
    python_requires='>=3.5.0',)

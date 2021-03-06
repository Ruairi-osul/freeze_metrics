from setuptools import setup, find_packages

with open("requirements.txt") as f:
    reqs = f.read().splitlines()

description = ""
long_description = ""
setup(
    name="freeze_metrics",
    description=description,
    long_description=long_description,
    version="0.0.1",
    url="https://github.com/Ruairi-osul/freeze_metrics",
    author="Ruairi O'Sullivan",
    author_email="ruairi.osullivan.work@gmail.ie",
    license="GNU GPLv3",
    keywords="neuroscience",
    project_urls={"Source": "https://github.com/Ruairi-osul/freeze_metrics"},
    packages=find_packages(),
    python_requires=">=3.3",
    install_requires=reqs,
)

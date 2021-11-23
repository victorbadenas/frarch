#!/usr/bin/env python
import glob
import os

from setuptools import find_packages, setup


def read(fname):
    """
    Read the contents of a file.

    Parameters
    ----------
    fname : str
        Path to file.

    Returns
    -------
    str
        File contents.
    """
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


install_requires = read("requirements.txt").splitlines()

extras_require = {}
extra_req_files = glob.glob("requirements-*.txt")
for extra_req_file in extra_req_files:
    name = os.path.splitext(extra_req_file)[0].replace("requirements-", "", 1)
    extras_require[name] = read(extra_req_file).splitlines()

if extras_require:
    extras_require["all"] = sorted({x for v in extras_require.values() for x in v})


meta = {}
exec(read("frarch/__meta__.py"), meta)

long_description = read("README.md")


setup(
    # Essential details on the package and its dependencies
    name=meta["name"],
    version=meta["version"],
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_dir={meta["name"]: os.path.join(".", meta["path"])},
    # If any package contains *.txt or *.rst files, include them:
    # package_data={"": ["*.txt", "*.rst"],}
    install_requires=install_requires,
    extras_require=extras_require,
    # Metadata to display on PyPI
    author=meta["author"],
    author_email=meta["author_email"],
    description=meta["description"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=meta["license"],
    url=meta["url"],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

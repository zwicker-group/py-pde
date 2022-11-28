from pathlib import Path

import versioneer
from setuptools import find_packages, setup

# determine the version of the package
version = versioneer.get_version()

# read the description from the README file
BASE_PATH = Path(__file__).resolve().parent
with open(BASE_PATH / "README.md", "r") as fh:
    long_description = fh.read()

setup(
    packages=find_packages(),
    include_package_data=True,  # include template files and the like
    zip_safe=False,  # this is required for mypy to find the py.typed file
    version=version,
    cmdclass=versioneer.get_cmdclass(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    download_url=f"https://github.com/zwicker-group/py-pde/archive/{version}.tar.gz",
)

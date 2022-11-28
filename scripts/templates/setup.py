import versioneer
from setuptools import find_packages, setup

# determine the version of the package
version = versioneer.get_version()

# most arguments for setup are defined in pyproject.toml
setup(
    packages=find_packages(),
    version=version,
    cmdclass=versioneer.get_cmdclass(),
    download_url=f"https://github.com/zwicker-group/py-pde/archive/{version}.tar.gz",
)

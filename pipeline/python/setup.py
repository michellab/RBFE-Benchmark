# setup.py
import setuptools

__version__ = "0.1"
VERSION = __version__
ISRELEASED = False

setuptools.setup(
    name="pipeline",
    version=__version__,
    author="Anna Herz",
    author_email="anna.herz@ed.ac.uk",
    description="Library for the setup and anlysis of RBFE calculations with AMBER, SOMD, and GROMACS using BSS",
    packages=setuptools.find_packages(),
    url="https://github.com/annamherz/code",
)

import codecs
import os
import sys

from setuptools import find_packages, setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = os.path.join(PROJECT_ROOT, "README.md")
VERSION_FILE = os.path.join(PROJECT_ROOT, "segmented", "version.py")
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")


def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


with open(VERSION_FILE) as buff:
    exec(buff.read())

if len(set(("test", "easy_install")).intersection(sys.argv)) > 0:
    import setuptools

tests_require=['pytest']
extra_setuptools_args = {}

setup(
    name="segmented",
    version=__version__,
    description="segmented: A toolbox for segmented regression.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="http://github.com/cluhmann/segmented",
    download_url="https://github.com/cluhmann/segmented/archive/%s.tar.gz" % __version__,
    install_requires=get_requirements(),
    maintainer="Christian Luhmann <cluhmann@gmail.com>",
    maintainer_email="cluhmann@gmail.com",
    packages=find_packages(exclude=["tests", "test_*"]),
    tests_require=tests_require,
    test_suite='nose.collector',
    license="MIT",
    **extra_setuptools_args,
)

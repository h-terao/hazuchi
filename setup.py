from setuptools import setup, find_packages

AUTHOR = "TERAO Hayato"
AUTHOR_EMAIL = ""

URL = "https://github.com/h-terao/hazuchi"
LICENSE = "MIT License"
DOWNLOAD_URL = "hoge"


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="hazuchi",
    version="v0.0.1",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description="Hazuchi: An all-in-one JAX/Flax training wrapper.",
    long_description=open("README.md").read(),
    license="MIT LICENSE",
    url=URL,
    download_url=DOWNLOAD_URL,
    python_requires=">=3.7",
    install_requires=_requires_from_file("requirements.txt"),  # INSTALL_REQUIRES,
    extras_require={
        "dev": [
            "flake8",
            "black",
        ],
    },
    packages=find_packages(),  
    package_dir={"": "hazuchi"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
)

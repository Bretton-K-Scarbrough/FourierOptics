from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Library for Fourier Optics Helper Functions"

# Setting up
setup(
    name="fourieroptics",
    version=VERSION,
    author="Bretton Scarbrough",
    author_email="<bretton.scarbrough@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["numpy", "scipy"],
    keywords=[
        "python",
        "fourier",
        "optics",
        "fourier optics",
    ],
    classifiers=[
        "Development Status :: 1 - Devloping",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bkm-engine",
    version="2.0.0",
    author="Joshua Daniel Curry",
    author_email="jcurry3428@smail.pcd.edu",
    description="GPU-accelerated pseudo-spectral solver for 3D incompressible Navier-Stokes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JDCurry/bkm-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "h5py>=3.0.0",
    ],
    extras_require={
        "gpu": ["cupy-cuda11x>=10.0.0"],
        "analysis": ["matplotlib>=3.3.0", "pandas>=1.3.0"],
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9"],
    },
)
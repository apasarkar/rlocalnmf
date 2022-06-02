import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="rlocalnmf",
    version="0.0.4",
    description="New implementation of localnmf with advanced background models and initialization options",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy","cvxpy","Cython", "networkx","scikit-learn", "torch", "matplotlib", "opencv-python", "scikit-image"],
    classifiers=(
        "Programming Language :: Python :: 3",
    ),
)
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rlocalnmf",
    version="0.0.4",
    description="New implementation of localnmf with advanced background models and initialization options",
    packages=setuptools.find_packages(),
    install_requires=["torch>=2.0.0", "torchvision", "torchaudio", "numpy", "scipy", "cvxpy", "Cython",
                      "networkx", "jupyterlab",
                      "scikit-learn", "matplotlib",
                      "opencv-python", "scikit-image", "tqdm", "oasis-deconv"],
    classifiers=(
        "Programming Language :: Python :: 3",
    ),
)

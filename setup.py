from setuptools import find_packages, setup

setup(
    name="src",
    version="0.1.0",
    description=(
        "Adversarial Robustness for Speaker Recognition"
        "This is a sub-project of the COMFORT project."
    ),
    author="Ahmad Aloradi",
    author_email="ahmad.aloradi94@gmail.com",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(exclude=["tests"]),
)

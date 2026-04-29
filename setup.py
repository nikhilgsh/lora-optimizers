from setuptools import find_packages, setup

setup(
    name="lora-playground",
    version="0.1",
    description="Speedrun-style LoRA optimizer training playground",
    package_dir={"": "."},
    packages=find_packages(include=["lora_playground", "lora_playground.*"]),
    python_requires=">=3.10",
)

from setuptools import setup, find_packages
import os

# 明确指定包的位置和子包
packages = [
    "cod",
    "cod.models", 
    "cod.training",
    "cod.evaluation",
    "cod.utils"
]
print(f"Explicitly defined packages: {packages}")

setup(
    name="cod",
    version="0.2.0",
    description="Chain of Debate (CoD) - Reasoning Model",
    packages=packages,
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "tokenizers>=0.12.0",
        "accelerate>=0.20.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
) 
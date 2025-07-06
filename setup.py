from setuptools import setup, find_packages

setup(
    name="ksf",
    version="2.1.0",
    description="Knowledge Synthesized Framework (KSF) - V2",
    # Use find_packages to automatically discover all packages under the 'ksf' directory.
    packages=find_packages(where=".", include=['ksf', 'ksf.*']),
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
        "sentence_transformers" # Added for knowledge injection
    ],
) 
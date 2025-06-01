#!/usr/bin/env python
"""Setup script for MemOLLM"""

from setuptools import setup, find_packages

setup(
    name="memollm",
    version="0.1.0",
    description="Memory-enhanced Language Learning Model with Ebbinghaus forgetting curve",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "matplotlib",
        "psutil",
    ],
    entry_points={
        'console_scripts': [
            'memollm-inference=scripts.inference:main',
            'memollm-test=scripts.test:main',
            'memollm-demo=scripts.demo:main',
        ],
    },
)
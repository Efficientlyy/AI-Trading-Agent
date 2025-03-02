#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

# Check if Rust toolchain is available
try:
    import subprocess
    has_rust = subprocess.call(
        ["rustc", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    ) == 0
except Exception:
    has_rust = False

if not has_rust and not os.environ.get("SKIP_RUST_CHECK"):
    print("⚠️  Rust not found. Project will be installed without Rust components.")
    print("📝 To install Rust, visit: https://rustup.rs/")
    print("🔄 To skip this check, set SKIP_RUST_CHECK=1")
    rust_extensions = []
else:
    rust_extensions = [
        RustExtension(
            "crypto_trading_engine",
            "rust/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
            features=["python"],
        )
    ]

# Package metadata
NAME = "ai_crypto_trading"
DESCRIPTION = "AI-powered cryptocurrency trading system with high-performance Rust components"
URL = "https://github.com/yourusername/ai-crypto-trading-system"
EMAIL = "your.email@example.com"
AUTHOR = "Your Name"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.1.0"

# Required packages
REQUIRED = [
    "numpy>=1.20.0",
    "pandas>=1.3.0", 
    "matplotlib>=3.4.0",
    "scipy>=1.7.0",
    "structlog>=21.1.0",
    "pydantic>=1.9.0",
    "pyyaml>=6.0.0",
]

# Optional packages
EXTRAS = {
    "dev": [
        "pytest>=6.2.5",
        "black>=21.5b2", 
        "mypy>=0.910",
        "flake8>=4.0.0",
        "isort>=5.9.0",
    ],
    "ml": [
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "torch>=1.10.0",
    ],
    "viz": [
        "plotly>=5.3.0",
        "dash>=2.0.0",
    ],
}

# Where the magic happens
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Rust",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    rust_extensions=rust_extensions,
    zip_safe=False,  # Required for Rust extensions
    entry_points={
        "console_scripts": [
            "crypto-trader=src.main:main",
            "crypto-backtest=examples.backtest_example:main",
        ],
    },
) 
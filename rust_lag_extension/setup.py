from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="rust_lag_extension",
    version="0.1.0",
    description="Rust-accelerated lag features for time series analysis",
    packages=["python_package"],
    package_dir={"python_package": "python_package"},
    rust_extensions=[
        RustExtension(
            "rust_lag_extension",
            path="Cargo.toml",
            binding=Binding.PyO3,
            features=["extension-module"],
        )
    ],
    # rust extensions are not zip safe, just like C-extensions
    zip_safe=False,
    install_requires=[
        "numpy",
    ],
)

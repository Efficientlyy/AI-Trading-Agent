from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="rust_test_extension",
    version="0.1.0",
    rust_extensions=[
        RustExtension(
            "rust_test_extension",
            path="Cargo.toml",
            binding=Binding.PyO3,
            features=["extension-module"],
        )
    ],
    # rust extensions are not zip safe, just like C-extensions
    zip_safe=False,
)

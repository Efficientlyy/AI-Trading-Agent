from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="ai_trading_agent_rs",
    version="0.1.0",
    rust_extensions=[
        RustExtension(
            "ai_trading_agent.rust_extensions",
            path="rust_extensions/Cargo.toml",
            binding=Binding.PyO3,
            features=["extension-module"],
        )
    ],
    packages=["ai_trading_agent"],
    # rust extensions are not zip safe, just like C-extensions
    zip_safe=False,
)

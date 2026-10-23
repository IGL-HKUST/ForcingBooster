from setuptools import find_packages, setup


setup(
    name="causal-forcing",
    version="0.1.0",
    description="Multi-model causal video distillation workspaces",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "methods": [
            "Wan21/*/README.md",
            "Wan21/*/manifest.yaml",
            "Wan21/*/config/*.yaml",
            "Wan21/*/runtime/*.yaml",
            "Wan21/*/scripts/*.sh",
            "LingBotWorld/*/README.md",
            "LingBotWorld/*/config/*.yaml",
            "LingBotWorld/*/runtime/*.yaml",
            "LingBotWorld/*/scripts/*.sh",
            "LingBotWorld/**/*.txt",
            "H3/*/LICENSE",
            "H3/*/NOTICE",
            "H3/*/requirements.txt",
            "H3/*/config/*.yaml",
            "H3/*/scripts/*.sh",
        ]
    },
    python_requires=">=3.10",
    install_requires=["omegaconf>=2.3.0"],
    entry_points={"console_scripts": ["causal-forcing=methods.cli:main"]},
)

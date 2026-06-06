from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def read_long_description() -> str:
    readme = ROOT / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return "Molecular modelling and crystallography utilities."


setup(
    name="molib",
    version="0.1.0",
    description="Molecular modelling and crystallography utilities.",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["molib", "molib.*"]),
    include_package_data=True,
    package_data={"molib": ["**/*.md"]},
    python_requires=">=3.10",
    install_requires=[
        "biopandas",
        "biopython",
        "decologr",
        "gemmi",
        "numpy",
        "pandas",
        "plotly",
        "PyOpenGL",
        "rdkit",
        "scikit-image",
        "scipy",
    ],
)

from setuptools import find_packages, setup

install_requires = [
    "datasets",
    "einops",  # Test dependency
    "fancy_einsum",  # Test dependency
    "jaxtyping",  # Test dependency
    "kaleido",
    "line_profiler",
    "matplotlib",
    "numpy",  # Test dependency
    "open-clip-torch",
    "plotly==5.19.0",
    "pytest>=6.0",  # Test dependency, but included here to auto-install
    "pre-commit",
    "scikit-learn",  # Test dependency
    "timm",  # Test dependency
    "torch",  # Test dependency
    "transformers",  # Test dependency
    "wandb",
]

setup(
    name="vit-prisma",
    version="2.0.0",
    author="Sonia Joseph",
    author_email="soniamollyjoseph@gmail.com",
    description="A Vision Transformer library for mechanistic interpretability.",
    long_description=open("docs/README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/soniajoseph/vit-prisma",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "vit_prisma": ["visualization/*.html", "visualization/*.js"],
        # Add other patterns here as needed
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
    keywords="vision-transformer, clip, multimodal, machine-learning, mechanistic "
    "interpretability",
    zip_safe=False,
    extras_require={
        "sae": ["sae-lens==2.1.3"],
        "arrow": ["pyarrow"],
        # to use: pip install -e .[sae] # as of 2.1.3, windows will requires:
        # pip install sae-lens==2.1.3 --no-dependencies followed by manually installing
        # needed packages
    },
)

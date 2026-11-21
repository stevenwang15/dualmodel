from setuptools import setup, find_packages

setup(
    name="dualmodel",
    version="0.2.0",
    description="DualModel - 双模型协作推理框架",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="DualModel Contributors",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "openai>=1.6.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "train": [
            "sentence-transformers>=2.2.0",
            "scikit-learn>=1.3.0",
            "modelscope>=1.0.0",
        ],
        "api": ["fastapi>=0.104.0", "uvicorn>=0.24.0"],
        "tray": [
            "rumps>=0.4.0; sys_platform == 'darwin'",
            "pystray>=0.19.0; sys_platform == 'win32'",
            "Pillow>=9.0.0; sys_platform == 'win32'",
        ],
        "all": [
            "sentence-transformers>=2.2.0",
            "scikit-learn>=1.3.0",
            "modelscope>=1.0.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dualmodel=dualmodel.__main__:main",
            "dualmodel-api=dualmodel.api_server:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

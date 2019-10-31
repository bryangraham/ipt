import setuptools

with open("readme.txt", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ipt",
    version="0.0.1",
    author="Bryan Graham",
    description="a Python 3.7 package for causal inference by inverse probability tilting",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/bryangraham/ipt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="detectshift",
    version="1.0.0",
    author="Felipe Maia Polo",
    author_email="felipemaiapolo@gmail.com",
    description="DetectShift: A unified framework for dataset shift diagnostics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/felipemaiapolo/detectshift',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy','pandas','sklearn','tqdm','catboost'],
) 
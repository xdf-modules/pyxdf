from pytest import fixture


@fixture(scope="session")
def example_files():
    from pathlib import Path

    # requires git clone https://github.com/xdf-modules/example-files.git
    # into the root xdf-python folder
    path = Path("example-files")
    extensions = ["*.xdf", "*.xdfz", "*.xdf.gz"]
    files = []
    for ext in extensions:
        files.extend(path.glob(ext))
    files = [str(file) for file in files]
    yield files

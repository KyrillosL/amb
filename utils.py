from pathlib import Path

allowed_extensions = ['.wav']  # in any case we want to add other formats one day


# Filter not allowed extension (and remove any annoying '.DS_STORE')
def filter_files(files: list[Path]) -> list[Path]:
    out = []
    for file in files:
        if file.suffix in allowed_extensions:
            out.append(file)
    return out


def get_files(path) -> list[Path]:
    files = [child.resolve() for child in Path.iterdir(Path(path))]
    # Filter out files that doesn't match allowed extensions
    files = filter_files(files)  # shadowing 'files'
    # Don't need to sort, but I prefer to the files alphabetically
    files.sort()
    return files


if __name__ == "__main__":
    for file in get_files():
        print(file)

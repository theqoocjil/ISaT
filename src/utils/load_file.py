import pandas as pd

def load_file(file_path: str, sep: str = ";") -> pd.DataFrame:
    """
    Returns the DataFrame of the read file 

    Args:
        file_path: The path to the dataset file
        sep: Character or regex pattern to treat as the delimiter
    """
    df = pd.read_csv(filepath_or_buffer=file_path, sep=sep)

    return df
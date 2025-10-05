import pandas as pd

def load_file(file_path: str) -> pd.DataFrame:
    """
    Returns the DataFrame of the read file 

    Args:
        file_path: The path to the dataset file
    """
    df = pd.read_csv(filepath_or_buffer=file_path)

    return df
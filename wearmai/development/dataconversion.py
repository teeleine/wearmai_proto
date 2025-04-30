import pandas as pd
import os



def txt_to_df(filepath):
    """
    This function reads a text file and converts it to a pandas dataframe.
    :param filepath: The path to the text file.
    :return: A pandas dataframe.
    """
    # Read the text file
    with open(filepath, 'r') as file:
        data = file.read()

    # Split the text file into a list of lines
    lines = data.split('\n')

    # Split the lines into a list of lists
    data = [line.split() for line in lines]

    # Convert the list of lists to a pandas dataframe
    df = pd.DataFrame(data[1:], columns=data[0])

    return df


import numpy as np

def get_summary(list, percision = 4):
    """
    This function returns the summary of a list.
    :param list: The list to get the summary of.
    :return: The summary of the list as a dictionary.
    """
    summary = {
        'min': round(min(list), percision),
        'q1': round(np.percentile(list, 25), percision),
        'median': round(np.percentile(list, 50), percision),
        'q3': round(np.percentile(list, 75), percision),
        'max': round(max(list), percision),
        'mean': round(np.mean(list), percision),
        'std': round(np.std(list), percision)
    }

    # Turn all into floats
    for key, value in summary.items():
        summary[key] = float(value)
    
    return summary
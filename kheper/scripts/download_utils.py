"""
Infixion Bioscience
Download Utility Functions
Gregory Way 2019

Helper functions to prioritize NF1 function altering compounds
"""

import os


def download_data(base_url, name, folder, append_url=True):
    """
    Download data from the web given a url and a destination folder

    Arguments:
    base_url - the basename of the url to retrieve data from
    name - the name of the file
    folder - the output destination folder of the file
    append_url - boolean if the name should be appended to the base_url
    """
    from urllib.request import urlretrieve

    path = os.path.join(folder, name)
    os.makedirs(folder, exist_ok=True)

    if append_url:
        base_url = '{}{}'.format(base_url, name)

    urlretrieve(base_url, path)

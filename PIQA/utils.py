import sys

sys.path.append(".")

import pickle


def save_dictionary(dictionary, save_path):
    """
    This method is used to save dictionary to a given path in pickle file
    Args:
        dictionary (dict): dictionary which has to be saved
        save_path (str): path where the dictionary has to be saved
    """
    with open(save_path, 'wb') as handle:
        print("saving model to:", save_path)
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("dictionary saved to:", save_path)


def load_dictionary(load_path):
    """
    This method is used to load the dictionary given a path
    Args:
        load_path (str): loading dictionary from the given path

    Returns:
        dictionary (dict): the loaded dictionary

    """
    with open(load_path, 'rb') as handle:
        print("loading data from:", load_path)
        dictionary = pickle.load(handle)
        print("loading completed")
        return dictionary

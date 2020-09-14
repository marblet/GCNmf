import torch


def generate_mask(features, missing_rate, missing_type):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float
    missing_type : string

    Returns
    -------

    """
    if missing_type == 'random':
        return generate_random_mask(features, missing_rate)
    if missing_type == 'struct':
        return generate_struct_mask(features, missing_rate)
    raise ValueError("Missing type {0} is not defined".format(missing_type))


def generate_random_mask(features, missing_rate):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float

    Returns
    -------
    mask : torch.tensor
        mask[i][j] is True if features[i][j] is missing.

    """
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask


def generate_struct_mask(features, missing_rate):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float

    Returns
    -------
    mask : torch.tensor
        mask[i][j] is True if features[i][j] is missing.

    """
    node_mask = torch.rand(size=(features.size(0), 1))
    mask = (node_mask <= missing_rate).repeat(1, features.size(1))
    return mask


def apply_mask(features, mask):
    """

    Parameters
    ----------
    features : torch.tensor
    mask : torch.tensor

    """
    features[mask] = float('nan')

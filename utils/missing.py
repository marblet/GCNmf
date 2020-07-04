import torch


def generate_mask(features, missing_rate, missing_type):
    if missing_type == 'random':
        return generate_random_mask(features, missing_rate)
    elif missing_type == 'struct':
        return generate_struct_mask(features, missing_rate)
    else:
        raise ValueError("Missing type {0} is not defined".format(missing_type))


def generate_random_mask(features, missing_rate):
    mask = torch.rand(size=features.size())
    return mask <= missing_rate


def generate_struct_mask(features, missing_rate):
    node_mask = torch.rand(size=(features.size(0), 1))
    mask = (node_mask <= missing_rate).repeat(1, features.size(1))
    return mask


def apply_mask(features, mask):
    features[mask] = float('nan')

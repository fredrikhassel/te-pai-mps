"""PAI probability decomposition functions for TE-PAI."""

import numpy as np

def prob_list(angles, delta):
    probs = [abc(theta, (1 if theta >= 0 else -1) * delta) for theta in angles]
    return [list(np.abs(probs) / np.sum(np.abs(probs))) for probs in probs]

def abc(theta, delta):
    a = (1 + np.cos(theta) - (np.cos(delta) + 1) / np.sin(delta) * np.sin(theta)) / 2
    b = np.sin(theta) / np.sin(delta)
    c = (1 - np.cos(theta) - np.sin(theta) * np.tan(delta / 2)) / 2
    return np.array([a, b, c])

def gamma(angles, delta):
    gam = [
        np.cos(np.sign(theta) * delta / 2 - theta) / np.cos(np.sign(theta) * delta / 2)
        for theta in angles
    ]
    return np.prod(np.array(gam))
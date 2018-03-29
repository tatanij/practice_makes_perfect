# l2 reg uses gaussian distribution and a gaussian prior on w 

import numpy as np

def l2_reg(l, w, dy=False):
    if dy:
        return l*w
    else:
        return (l/2)*w.T.dot(w)
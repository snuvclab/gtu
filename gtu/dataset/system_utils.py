import os

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    saved_iters = sorted(saved_iters)

    # skip 6000 iters
    idx = -1
    while (saved_iters[idx] in [6000, 6010]):
        idx -= 1
    
    return saved_iters[idx]

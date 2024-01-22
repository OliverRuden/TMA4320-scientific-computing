import numpy as np


"""
1 b) Her lager vi polymeret vårt, det er representert i et N*2 diagram, og vi har valgt å holde fast opprunnet midten hvis relevant
"""
def createPolymer(N):
    polymer = np.zeros((N,2)) 
    polymer[:,1] = N//2         # setter y-koordinat
    for i in range(N):
        polymer[i,0] = i        # setter x-koordinat
    return polymer
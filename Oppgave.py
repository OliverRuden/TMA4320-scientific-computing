import numpy as np


"""
1 b) Her lager vi polymeret vårt, det er representert i et N*2 diagram, og vi har valgt å holde fast opprunnet midten hvis relevant
"""
def createPolymer(N):
    array = np.zeros((N,2))
    array[:,1] = N//2
    for i in range(N):
        array[i,0] = i
    return array

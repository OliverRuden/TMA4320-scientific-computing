import numpy as np
import matplotlib.pyplot as plt


"""
1 b) Her lager vi polymeret vårt, det er representert i et N*2 diagram, og vi har valgt å holde fast opprunnet midten hvis relevant
"""
def createPolymer(N):
    polymer = np.zeros((N, 2)) 
    polymer[:, 1] = N // 2                                     # setter y-koordinat
    polymer[:, 0] = np.array([i for i in range(N)])          # setter x-koordinat
    return polymer
"""
1 d) Her lager vi illustrasjon av polymeret vårt, det er representert med farge, der sterkere farge er monomer med høyere nummer
"""
def illustrationPolymer(polymer):
    N = len(polymer)                 
    grid = np.zeros((N+1,N+1))        # Lager (N+1)*(N+1) grid
    grid -= int(N/2)                         # Setter bakgrunnsverdien til å være -N for å få synlighet blant lave N
    for monomerNumber in range(N):
        x = int(polymer[monomerNumber, 0])  
        y = int(polymer[monomerNumber, 1])
        grid[y,x] = monomerNumber + 1
    plt.pcolormesh(grid)
    plt.show()

"""
1 e) Sjekker om intakt polymer
"""

def validPolymer(polymer, N):
    if len(polymer) != N:
        return False
    
    coordinateSet = set()
    coordinateSet.add((polymer[0, 0], polymer[0, 1]))

    for index in range(1, N):
        if (polymer[index, 0], polymer[index, 1]) in coordinateSet:         # sjekker om andre monomer har samme koordinat
            return False
        else: 
            coordinateSet.add((polymer[index, 0], polymer[index, 1]))

        xDiff = np.abs(polymer[index, 0] - polymer[index - 1, 0])           # avstand i x til nabo-monomer
        yDiff = np.abs(polymer[index, 1] - polymer[index - 1, 1])           # avstand i y til nabo-monomer

        if xDiff + yDiff != 1:                                              
            return False
        
    return True

"""
1 f) Implementerer rotasjon ov polymeret, som nevnt tidligere holdes opprundet midten fast.
"""
def rotationGoBrrrr(polymer, monomer, positivRetning):
    middleMonomer = len(polymer)//2 #Finner midterste rundet opp, for å låse den...
    if middleMonomer > monomer:
        polymer[:monomer,0] = (2*positivRetning-1)*polymer[:monomer,1]
        polymer[:monomer,1] = (1-2*positivRetning)*polymer[:monomer,0]
        """
        Positiv retning:
        x = y
        y = -x
        Negativ retning:
        x = -y
        y = x

        Bruker også at True kan brukes som 1 og False som 0
        """
        return polymer

    polymer[monomer+1:,0] = (1-2*positivRetning)*polymer[monomer+1:,1]
    polymer[monomer+1:,1] = (2*positivRetning-1)*polymer[monomer+1:,0]
    """
        Positiv retning:
        y = x
        x = -y
        Negativ retning:
        y = -x
        x = y
    """
    return polymer
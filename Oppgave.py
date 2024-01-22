import numpy as np
import matplotlib.pyplot as plt


"""
1 b) Her lager vi polymeret vårt, det er representert i et N*2 diagram, og vi har valgt å holde fast opprunnet midten hvis relevant
"""
def createPolymer(N):
    polymer = np.zeros((N,2)) 
    polymer[:,1] = N//2                                     # setter y-koordinat
    polymer[:,0] = np.array([i for i in range(N)])          # setter x-koordinat
    return polymer
"""
1 d) Her lager vi illustrasjon av polymeret vårt, det er representert med farge, der sterkere farge er monomer med høyere nummer
"""
def illustrationPolymer(polymer):
    N=len(polymer)                 
    grid=np.zeros((N+1,N+1))        # Lager (N+1)*(N+1) grid
    grid-=int(N/2)                         # Setter bakgrunnsverdien til å være -N for å få synlighet blant lave N
    for monomerNumber in range(N):
        x=int(polymer[monomerNumber,0])  
        y=int(polymer[monomerNumber,1])
        grid[y,x]=monomerNumber+1
    plt.pcolormesh(grid)
    plt.show()

N=1000
polymer=createPolymer(N)
illustrationPolymer(polymer)
    
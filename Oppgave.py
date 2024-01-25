import numpy as np
import matplotlib.pyplot as plt
import timeit


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
    monomer -= 1
    middleMonomer = len(polymer)//2 #Finner midterste rundet opp, for å låse den...
    x,y = polymer[monomer]
    newPolymer = np.zeros((len(polymer),2)) #Lager et nytt polymer, fordi å jobbe inplace endret på dataen underveis
    if middleMonomer > monomer:
        newPolymer[monomer:] = polymer[monomer:]
        newPolymer[:monomer,0] = (2*positivRetning-1)*(polymer[:monomer,1]-y)+x
        newPolymer[:monomer,1] = (1-2*positivRetning)*(polymer[:monomer,0]-x) + y
        """
        Positiv retning:
        delta x = delta y
        delta y = - delta x
        Negativ retning:
        delta x = - delta y
        delta y = delta x

        Bruker også at True kan brukes som 1 og False som 0
        """
        return newPolymer
    newPolymer[:monomer+1] = polymer[:monomer+1]
    newPolymer[monomer+1:,0] = (1-2*positivRetning)*(polymer[monomer+1:,1]-y)+x
    newPolymer[monomer+1:,1] = (2*positivRetning-1)*(polymer[monomer+1:,0]-x)+y
    """
        Positiv retning:
        delta y = delta x
        delta x = - delta y
        Negativ retning:
        delta y = - delta x
        delta x = delta y
    """
    return newPolymer

def rotateManyTimes(N, Ns):
    rotationsMade = 0
    polymer = createPolymer(N)

    for i in range(Ns):
        monomer = np.random.randint(2, N)
        positivRetning = np.random.choice([True, False])

        twistedPolymer = rotationGoBrrrr(polymer, monomer, positivRetning)
        if validPolymer(twistedPolymer, N):
            rotationsMade += 1
            polymer = twistedPolymer

    return polymer, rotationsMade


"""
Oppgave g)
"""
pol_4, rot_4 = rotateManyTimes(15, 4)
print("Med 4 rotasjoner ble så mange gyldige: ", rot_4)
illustrationPolymer(pol_4)
np.savetxt('polymerArray15_4.txt', pol_4)
pol_1000, rot_1000 = rotateManyTimes(15,1000)
print("Med 1000 rotasjoner ble så mange gyldige: ", rot_1000)
illustrationPolymer(pol_1000)
np.savetxt('polymerArray15_1000.txt', pol_1000)

"""
1 i)
plotte valid rotations
"""

def plotValidPercentage(min = 4, max = 500, Ns = 1000):
    sizes = np.arange(min, max + 1, 10)
    intSizes = sizes.astype(int)
    print(intSizes[0])
    # polymer, validRot = rotateManyTimes(intSizes, Ns)
    valid = np.array([rotateManyTimes(i, Ns)[1] for i in intSizes])
    plt.plot(intSizes, valid/Ns)
    plt.show()

# plotValidPercentage(10, 500)
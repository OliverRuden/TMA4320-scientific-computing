import numpy as np
import matplotlib.pyplot as plt
import timeit


class Middle:
    def __init__(self, x, y, N):
        self.position = np.array([x,y])
        self.array = np.array([np.array((0,2)) for i in range(N)])
        self.array[0][0] = -1
        self.array[N-1][1] = -1
        self.map = {0:np.array([0,-1]), 1:np.array([1,0]), 2:np.array([0,1]),3:np.array([-1,0])}

"""
1 b) Her lager vi polymeret vårt, det er representert i et N*2 diagram, og vi har valgt å holde fast opprunnet midten hvis relevant
"""
def createPolymer(N):
    middle = Middle(N//2, N//2, N)    # setter x-koordinat
    return middle
"""
1 d) Her lager vi illustrasjon av polymeret vårt, det er representert med farge, der sterkere farge er monomer med høyere nummer
"""
def illustrationPolymer(polymer):
    N = len(polymer.array)                 
    grid = np.zeros((N+1,N+1))        # Lager (N+1)*(N+1) grid
    grid -= int(N/2)                         # Setter bakgrunnsverdien til å være -N for å få synlighet blant lave N
    index = N//2
    position = np.copy(polymer.position)
    direction = 0
    for firstMonomers in range(index,0,-1):
        direction = (direction + polymer.array[firstMonomers,0])%4
        position += polymer.map[direction]
        grid[position[0],position[1]] = firstMonomers
    direction = 0
    position = np.copy(polymer.position)
    grid[position[0],position[1]] = N//2 + 1
    for secondMonomers in range(index, N-1):
        direction = (direction + polymer.array[secondMonomers,1]-2)%4
        position += polymer.map[direction]
        grid[position[0],position[1]] = secondMonomers+2
    plt.pcolormesh(grid)
    plt.show()

"""
1 e) Sjekker om intakt polymer
"""

def validPolymer(polymer, N):
    if len(polymer.array) != N:
        return False
    
    coordinateSet = set()
    coordinateSet.add((polymer.position[0], polymer.position[1]))
    index = N//2
    position = np.copy(polymer.position)
    direction = 0
    for firstMonomers in range(index,0,-1):
        if polymer.array[firstMonomers][0] not in polymer.map:
            return False
        elif polymer.array[firstMonomers][1] not in polymer.map:
            return False
        direction = (direction + polymer.array[firstMonomers,0])%4
        position += polymer.map[direction]
        if (position[0],position[1]) in coordinateSet:
            return False
        else:
            coordinateSet.add((position[0],position[1]))
    direction = 0
    position = np.copy(polymer.position)
    for secondMonomers in range(index,N-1):
        if polymer.array[secondMonomers][0] not in polymer.map:
            return False
        elif polymer.array[secondMonomers][1] not in polymer.map:
            return False
        direction = (direction + polymer.array[secondMonomers,1]-2)%4
        position += polymer.map[direction]
        if (position[0],position[1]) in coordinateSet:
            return False
        else:
            coordinateSet.add((position[0],position[1]))
    return True


"""
1 f) Implementerer rotasjon ov polymeret, som nevnt tidligere holdes opprundet midten fast.
"""
def rotationGoBrrrr(polymer, monomer, positivRetning):
    monomer -= 1
    middleMonomer = len(polymer.array)//2 #Finner midterste rundet opp, for å låse den...
    if middleMonomer > monomer:
        polymer.array[monomer,0] = (polymer.array[monomer,0] + 2*positivRetning-1) % 4
        """
        Positiv retning:
        delta x = delta y
        delta y = - delta x
        Negativ retning:
        delta x = - delta y
        delta y = delta x

        Bruker også at True kan brukes som 1 og False som 0
        """
        return polymer
    polymer.array[monomer,1] = (polymer.array[monomer,1] - 2*positivRetning+1) % 4
    """
        Positiv retning:
        delta y = delta x
        delta x = - delta y
        Negativ retning:
        delta y = - delta x
        delta x = delta y
    """
    return polymer

def rotateManyTimes(N, Ns):
    rotationsMade = 0
    polymer = createPolymer(N)

    for i in range(Ns):
        monomer = np.random.randint(2, N)
        positivRetning = np.random.randint(0,2)

        polymer = rotationGoBrrrr(polymer, monomer, positivRetning)
        if validPolymer(polymer, N):
            rotationsMade += 1
        else:
            polymer = rotationGoBrrrr(polymer,monomer,(positivRetning+1)%2)

    return polymer, rotationsMade


"""
Oppgave g)
"""
def saveTwoPolymers():
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

#print(timeit.timeit('rotateManyTimes(150,10000)', "from __main__ import rotateManyTimes", number = 10))

pol, rot = rotateManyTimes(10,1000)
print(rot)
illustrationPolymer(pol)

plotValidPercentage(10, 500)
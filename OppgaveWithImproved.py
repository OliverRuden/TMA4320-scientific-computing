import numpy as np
import matplotlib.pyplot as plt
import timeit


class Middle:
    def __init__(self, x, y, N):
        self.position = np.array([x,y])
        self.beforeMiddle = np.array([0 for i in range(N//2)])
        self.afterMiddle = np.array([2 for i in range((N-1)//2)])
        self.map = {0:np.array([0,-1]), 1:np.array([1,0]), 2:np.array([0,1]),3:np.array([-1,0])}

"""
1 b) Her lager vi polymeret vårt, det er representert i et N*2 diagram, og vi har valgt å holde fast opprunnet midten hvis relevant
"""
def createPolymerVer2(N):
    middle = Middle(N//2, N//2, N)    # setter x-koordinat
    return middle
"""
1 d) Her lager vi illustrasjon av polymeret vårt, det er representert med farge, der sterkere farge er monomer med høyere nummer
"""
def illustrationPolymerVer2(polymer):
    N = len(polymer.beforeMiddle) + len(polymer.afterMiddle) + 1        
    grid = np.zeros((N+1,N+1))        # Lager (N+1)*(N+1) grid
    grid -= int(N/2)                         # Setter bakgrunnsverdien til å være -N for å få synlighet blant lave N
    index = N//2
    position = np.copy(polymer.position)
    direction = 0
    for firstMonomers in range(index-1,-1,-1):
        direction = (direction + polymer.beforeMiddle[firstMonomers])%4
        position += polymer.map[direction]
        grid[position[0],position[1]] = firstMonomers + 1
    direction = 2
    position = np.copy(polymer.position)
    grid[position[0],position[1]] = index + 1
    for secondMonomers in range(0, len(polymer.afterMiddle)):
        direction = (direction + polymer.afterMiddle[secondMonomers]-2)%4
        position += polymer.map[direction]
        grid[position[0],position[1]] = secondMonomers+ index + 2
    plt.pcolormesh(grid)
    plt.show()

"""
1 e) Sjekker om intakt polymer
"""

def validPolymerVer2(polymer, N):
    if len(polymer.beforeMiddle) + len(polymer.afterMiddle) + 1 != N:
        return False
    coordinateSet = set()
    coordinateSet.add((polymer.position[0], polymer.position[1]))
    index = N//2
    position = np.copy(polymer.position)
    direction = 0
    for firstMonomers in range(index-1,-1,-1):
        if polymer.beforeMiddle[firstMonomers] not in polymer.map:
            return False
        direction = (direction + polymer.beforeMiddle[firstMonomers])%4
        position += polymer.map[direction]
        if (position[0],position[1]) in coordinateSet:
            return False
        else:
            coordinateSet.add((position[0],position[1]))
    direction = 2
    position = np.copy(polymer.position)
    for secondMonomers in range(0,len(polymer.afterMiddle)):
        if polymer.afterMiddle[secondMonomers] not in polymer.map:
            return False
        direction = (direction + polymer.afterMiddle[secondMonomers]-2)%4
        position += polymer.map[direction]
        if (position[0],position[1]) in coordinateSet:
            return False
        else:
            coordinateSet.add((position[0],position[1]))
    return True

"""
1 f) Implementerer rotasjon ov polymeret, som nevnt tidligere holdes opprundet midten fast.
"""
def rotationGoBrrrrVer2(polymer, monomer, positivRetning):
    monomer -= 2
    middleMonomer = len(polymer.beforeMiddle)-1 #Finner midterste rundet opp, for å låse den...
    if middleMonomer > monomer:
        polymer.beforeMiddle[monomer] = (polymer.beforeMiddle[monomer] + 2*positivRetning-1) % 4
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
    polymer.afterMiddle[monomer-middleMonomer] = (polymer.afterMiddle[monomer-middleMonomer] - 2*positivRetning+1) % 4
    """
        Positiv retning:
        delta y = delta x
        delta x = - delta y
        Negativ retning:
        delta y = - delta x
        delta x = delta y
    """
    return polymer

def rotateManyTimesVer2(N, Ns):
    rotationsMade = 0
    polymer = createPolymerVer2(N)

    for i in range(Ns):
        monomer = np.random.randint(2, N)
        positivRetning = np.random.randint(0,2)

        polymer = rotationGoBrrrrVer2(polymer, monomer, positivRetning)
        if validPolymerVer2(polymer, N):
            rotationsMade += 1
        else:
            polymer = rotationGoBrrrrVer2(polymer,monomer,(positivRetning+1)%2)

    return polymer, rotationsMade


"""
Oppgave g)
"""
def saveTwoPolymersVer2():
    pol_4, rot_4 = rotateManyTimesVer2(15, 4)
    print("Med 4 rotasjoner ble så mange gyldige: ", rot_4)
    illustrationPolymerVer2(pol_4)
    np.savetxt('polymerArray15_4.txt', pol_4)
    pol_1000, rot_1000 = rotateManyTimesVer2(15,1000)
    print("Med 1000 rotasjoner ble så mange gyldige: ", rot_1000)
    illustrationPolymerVer2(pol_1000)
    np.savetxt('polymerArray15_1000.txt', pol_1000)

"""
1 i)
plotte valid rotations
"""

def plotValidPercentageVer2(min = 4, max = 500, Ns = 1000):
    sizes = np.arange(min, max + 1, 10)
    intSizes = sizes.astype(int)
    print(intSizes[0])
    # polymer, validRot = rotateManyTimes(intSizes, Ns)
    valid = np.array([rotateManyTimesVer2(i, Ns)[1] for i in intSizes])
    plt.plot(intSizes, valid/Ns)
    plt.show()

# print(timeit.timeit('rotateManyTimes(150,10000)', "from __main__ import rotateManyTimes", number = 10))

# pol, rot = rotateManyTimes(10,100000)
# print(rot)
# illustrationPolymer(pol)

# plotValidPercentage(10, 500)

"""
Regne ut energien til polymer
"""
def calculateEnergyVer2(polymer, V):
    total = 0
    neighbourDictionary = {}
    coordinates = np.copy(polymer.position)
    for neighbour in polymer.map.values():
        neighbourDictionary[tuple(coordinates + neighbour)] = [len(polymer.beforeMiddle)]
    direction = 0
    coordinates = np.copy(polymer.position)
    for firstMonomers in range(len(polymer.beforeMiddle)-1,-1,-1):
        direction = (direction + polymer.beforeMiddle[firstMonomers])%4
        coordinates += polymer.map[direction]
        cordTuple = tuple(coordinates)
        if cordTuple in neighbourDictionary:
            for nodes in neighbourDictionary[cordTuple]:
                total += V[nodes, firstMonomers]
        for neighbour in polymer.map.values():
            if tuple(coordinates + neighbour) in neighbourDictionary:
                neighbourDictionary[tuple(coordinates + neighbour)].append(firstMonomers)
            else:
                neighbourDictionary[tuple(coordinates + neighbour)] = [firstMonomers]
    direction = 2
    coordinates = np.copy(polymer.position)
    for secondMonomers in range(0, len(polymer.afterMiddle)):
        direction = (direction + polymer.afterMiddle[secondMonomers]-2)%4
        coordinates += polymer.map[direction]
        monomerNumber = len(polymer.beforeMiddle)+1+secondMonomers
        cordTuple = tuple(coordinates)
        if cordTuple in neighbourDictionary:
            for nodes in neighbourDictionary[cordTuple]:
                total += V[nodes, monomerNumber]
        for neighbour in polymer.map.values():
            if tuple(coordinates + neighbour) in neighbourDictionary:
                neighbourDictionary[tuple(coordinates + neighbour)].append(monomerNumber)
            else:
                neighbourDictionary[tuple(coordinates + neighbour)] = [monomerNumber]

    return total

def makeDiagonalForceArrayVer2(N, background_value):
    V = np.zeros((N+1,N+1))+background_value
    for i in range(N):
        V[i,i] = 0
        if i > 0:
            V[i,i-1] = 0
            V[i-1,i] = 0
    print(V)
    return V

N = 15
V = makeDiagonalForceArrayVer2(N, -4*10**(-21))
for i in range(10):
    pol, rot = rotateManyTimesVer2(N,1000)
    print(calculateEnergyVer2(pol, V))
    illustrationPolymerVer2(pol)
import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.spatial import ConvexHull

"""
Definerer viktige konstanter
"""
k_b=1.38*10**(-23)

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
        positivRetning = np.random.randint(0,2)

        twistedPolymer = rotationGoBrrrr(polymer, monomer, positivRetning)
        if validPolymer(twistedPolymer, N):
            rotationsMade += 1
            polymer = twistedPolymer

    return polymer, rotationsMade


# """
# 1 g)
# """
# pol_4, rot_4 = rotateManyTimes(15, 4)
# print("Med 4 rotasjoner ble så mange gyldige: ", rot_4)
# illustrationPolymer(pol_4)
# np.savetxt('polymerArray15_4.txt', pol_4)
# pol_1000, rot_1000 = rotateManyTimes(15,1000)
# print("Med 1000 rotasjoner ble så mange gyldige: ", rot_1000)
# illustrationPolymer(pol_1000)
# np.savetxt('polymerArray15_1000.txt', pol_1000)

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

"""
1 j)
Regne energien til et polymer
"""

def calculateEnergy(polymer, V):
    total = 0
    neighbourDictionary = {}
    direction = [[0,1],[0,-1],[1,0],[-1,0]]
    for index, coordinates in enumerate(polymer):
        cordTuple = (coordinates[0],coordinates[1])
        if cordTuple in neighbourDictionary:
            for n in neighbourDictionary[cordTuple]:
                total += V[index][n]
        for i in direction:
            temp = (coordinates[0]+i[0],coordinates[1]+i[1])
            if temp in neighbourDictionary:
                neighbourDictionary[temp].append(index)
            else:
                neighbourDictionary[temp] = [index]
    return total

def makeDiagonalForceArray(N, background_value):
    V = np.zeros((N,N))+background_value
    for i in range(N):
        V[i,i] = 0
        if i > 0:
            V[i,i-1] = 0
            V[i-1,i] = 0
    return V

# N = 15
# V = makeDiagonalForceArray(N, -4*10**(-21))
# pol, rot = rotateManyTimes(N,1000)
# illustrationPolymer(pol)
# print(calculateEnergy(pol, V))

"""
2 a)
Her implementerer vi metropolisalgoritmen, har også lagt inn calculateDiameterDepresso her fra 2g), siden den brukes i metropolisalgoritmen
"""

def calculateDiameterDepresso(polymer):
    maxDist = 0
    for i in range(len(polymer)):
        for j in range(i+1, len(polymer)):
            s = np.sum((polymer[i]-polymer[j])**2)
            if s > maxDist:
                maxDist = s
    return np.sqrt(maxDist)

# def calculateDiameterNotSoDepresso(polymer):                    #This is with convex hull, but doesn't seem to work cause the points are to colinear

#     # Compute the convex hull of the points
#     hull = ConvexHull(polymer)

#     # Initialize maximum distance to zero
#     max_distance = 0.0

#     # Iterate through all pairs of vertices in the convex hull
#     for i in range(len(hull.vertices)):
#         for j in range(i + 1, len(hull.vertices)):
#             distance = np.linalg.norm(polymer[hull.vertices[i]] - polymer[hull.vertices[j]])
#             if distance > max_distance:
#                 max_distance = distance

#     # Return the maximum distance
#     return max_distance

def metropolisalgoritmen(polymer, V, Ns, T, includeDiamter = False):
    E_array=np.zeros(Ns)
    E = calculateEnergy(polymer, V)
    if includeDiamter:
        d_array=np.zeros(Ns)
        d = calculateDiameterDepresso(polymer)
    i=0
    N=len(polymer)    
    beta = 1/(k_b*T)
    E_array[0]=E
    while i<Ns-1:
        newpolymer = rotationGoBrrrr(polymer, np.random.randint(2, N), np.random.randint(0,2))
        if validPolymer(newpolymer,N):
            i+=1
            E_new=calculateEnergy(newpolymer, V)
            if includeDiamter:
                d_new = calculateDiameterDepresso(newpolymer)
            if E_new < E:
                polymer = newpolymer
                E = E_new
                if includeDiamter:
                    d = d_new
            elif np.random.uniform() < np.exp(-beta*(E_new-E)):
                polymer = newpolymer
                E = E_new
                if includeDiamter:
                    d = d_new
            E_array[i] = E
            if includeDiamter:
                d_array[i] = d

    if includeDiamter:
        return polymer, E_array, d_array

    return polymer, E_array

"""
2 b) Plotte energien
"""

def plotEnergy(polymer, V, Ns, T):
    E_array = metropolisalgoritmen(polymer, V, Ns, T)[1]
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize = (10, 7))
    plt.plot(E_array, label = 'Energi')
    plt.xlabel(r'Monte Carlo-steg $t$')
    plt.ylabel('Energi')
    plt.title('Energi som funksjon av Monte Carlo-steg')
    plt.show()

N = 30
V = np.zeros((N,N))-4*10**(-21)
for i in range(N):
    V[i,i] = 0
    if i > 0:
        V[i,i-1] = 0
    if i < N-1:
        V[i+1,i] = 0

# plotEnergy(createPolymer(30), V, 5000, 370)

"""
2 c) Illustrere de to polymerene like
"""

def illustrationOfOnePolymer(polymer):              # Returnerer Grid
    N = len(polymer)                 
    grid = np.zeros((N+1,N+1))        # Lager (N+1)*(N+1) grid
    grid -= int(N/2)                         # Setter bakgrunnsverdien til å være -N for å få synlighet blant lave N
    for monomerNumber in range(N):
        x = int(polymer[monomerNumber, 0])  
        y = int(polymer[monomerNumber, 1])
        grid[y,x] = monomerNumber + 1        # Setter første monomer-verdi til å være 1
    return grid

def multiplePlotsPolymers(polymer1,polymer2, title1,title2):

    #Sublot 1
    grid_1 = illustrationOfOnePolymer(polymer1)
    plt.subplot(1,2,1)
    plt.title(title1)
    plt.pcolormesh(grid_1)

    #Subplot 2
    grid_2 = illustrationOfOnePolymer(polymer2)
    plt.subplot(1,2,2)
    plt.title(title2)
    plt.pcolormesh(grid_2)

    plt.show()

# V=makeDiagonalForceArray(30,-4*10**(-21))
# polymer_high_temp, E_array_high_temp=metropolisalgoritmen(createPolymer(30),V,5000,350)
# polymer_low_temp, E_array_low_temp=metropolisalgoritmen(createPolymer(30),V,5000,75)
# multiplePlotsPolymers(polymer_high_temp, polymer_low_temp, "High temperature polymer", "Low temperature polymer")

# illustrationPolymer(polymer)
# print(E_array[-1])

"""
2 d) 
"""
def createFunkyPotential(N, generalValue, scaling, tuplesToScale):
    potential = np.zeros((N,N))+generalValue
    for i in range(N):
        potential[i,i] = 0
        if i > 0:
            potential[i-1,i] = 0
            potential[i,i-1] = 0
    for tup in tuplesToScale:
        potential[tup] = generalValue*scaling
        potential[tup[1],tup[0]] = generalValue*scaling
    return potential
# N = 15
# V = createFunkyPotential(N,-4*10**(-21), 100, [(0,N-1),(1,N-2),(2,N-3),(3,N-4),(4,N-5),(N-1,N-4)])
# pol, array = metropolisalgoritmen(createPolymer(N), V, 100, 50)
# print(calculateEnergy(pol,V))
# print(min(array))
# illustrationPolymer(pol)

# print(timeit.timeit('rotateManyTimes(150,10000)', "from __main__ import rotateManyTimes", number = 10))

"""
2 e) Beregne forventningsverdi og standardavvik til energien
"""

def computeAverageEnergyAndSTD(V, T, Ns=1500, N=30):
    polymer = createPolymer(N)
    _, energy = metropolisalgoritmen(polymer, V, Ns, T)
    importantEnergy = energy[1000:]
    return np.average(importantEnergy), np.std(importantEnergy, ddof=1)

def plotExpectedAndSTDEnergy(V,lowTemp, highTemp, TempStep, Ns=1500, N=30):
    TempArray = np.arange(lowTemp,highTemp,TempStep)
    expectedValue, standardDeviation = np.zeros(len(TempArray)), np.zeros(len(TempArray))
    for temp_index in range(len(TempArray)):
        expectedValue[temp_index], standardDeviation[temp_index] = computeAverageEnergyAndSTD(V, TempArray[temp_index], Ns, N)
    plt.errorbar(TempArray, expectedValue, yerr = standardDeviation)
    plt.show()

# plotExpectedAndSTDEnergy(V, lowTemp=10,highTemp=1000,TempStep=30)
    
"""
2 f)
"""

def plotEnergyLowTemp(V, T, Ns = 1500, N = 30):
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize = (10, 7))
    for sim in range(10):
        polymer = createPolymer(N)
        _, energy = metropolisalgoritmen(polymer, V, Ns, T)
        plt.plot(energy)

    plt.xlabel(r'Monte Carlo-steg $t$')
    plt.ylabel('Energi')
    plt.title('Energi som funksjon av Monte Carlo-steg')
    plt.show()

# plotEnergyLowTemp(V, 20)
"""
2 g)
"""

def computeAverageDiameterAndSTD(V, T, Ns=1500, N=30):
    polymer = createPolymer(N)
    _,_,diameter = metropolisalgoritmen(polymer, V, Ns, T, includeDiamter=True)
    importantDiameter = diameter[1000:]
    return np.average(importantDiameter), np.std(importantDiameter, ddof=1)

def plotExpectedAndSTDDiameter(lowTemp, highTemp, TempStep, Ns=1500, N=30):
    
    V = np.zeros((N,N))
    for i in range(N):
        for j in range(i-1):
          V[i,j]=(np.random.uniform(-6,-2))*10**(-21)

    V=V+V.transpose()

    TempArray = np.arange(lowTemp,highTemp,TempStep)
    expectedValue, standardDeviation = np.zeros(len(TempArray)), np.zeros(len(TempArray))
    for temp_index in range(len(TempArray)):
        expectedValue[temp_index], standardDeviation[temp_index] = computeAverageDiameterAndSTD(V, TempArray[temp_index], Ns, N)
    plt.errorbar(TempArray, expectedValue, yerr = standardDeviation)
    plt.show()

# plotExpectedAndSTDDiameter(lowTemp=10, highTemp=1000, TempStep=100, Ns=1500, N=10)

print(timeit.timeit('plotExpectedAndSTDDiameter(lowTemp=10, highTemp=1000, TempStep=30, Ns=2000, N=30)', "from __main__ import plotExpectedAndSTDDiameter", number = 1))

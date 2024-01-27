import numpy as np
import matplotlib.pyplot as plt
import timeit

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
# Oppgave g)
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
        if i < N-1:
            V[i+1,i] = 0
    return V

# N = 1000
# V = makeDiagonalForceArray(N, -4*10**(-21))
# pol, rot = rotateManyTimes(15,1000)
# illustrationPolymer(pol)
# print(calculateEnergy(pol, V))

"""
2 a)
Her implementerer vi metropolisalgoritmen
"""

def metropolisalgoritmen(polymer, V, Ns, T):
    E_array=np.zeros(Ns)
    E = calculateEnergy(polymer, V)
    i=0
    N=len(polymer)    
    beta = 1/(k_b*T)
    E_array[0]=E
    while i<Ns-1:
        newpolymer = rotationGoBrrrr(polymer, np.random.randint(2, N), np.random.randint(0,2))
        if validPolymer(newpolymer,N):
            i+=1
            E_new=calculateEnergy(newpolymer, V)
            if E_new < E:
                polymer = newpolymer
                E = E_new
            elif np.random.uniform() < np.exp(-beta*(E_new-E)):
                polymer = newpolymer
                E = E_new
            E_array[i] = E
    return polymer, E_array

"""
2b) Plotte energien
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

# N = 30
# V = np.zeros((N,N))-4*10**(-21)
# for i in range(N):
#     V[i,i] = 0
#     if i > 0:
#         V[i,i-1] = 0
#     if i < N-1:
#         V[i+1,i] = 0

# plotEnergy(createPolymer(30), V, 5000, 370)

"""
2c) Illustrere de to polymerene like
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

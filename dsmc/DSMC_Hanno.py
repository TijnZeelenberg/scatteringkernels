import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jax import jit, lax, vmap
import math
import random
import timeit
import statistics



# Enable double precision in JAX
#jax.config.update("jax_enable_x64", True)

R = 8.31446261815324
kB = 1.380649 * 10e-23 #J/K
T = 6 #K
T1= 2
T2= 10
p = 100_000 #N/m^2
d = [289e-12, 364e-12, 346e-12] #m
chi = 2 #rigid rotor in BL (internal degrees of freedom?)
labda = (kB * T)/(math.sqrt(2) * math.pi * d[1] * d[1] * p)
Kn = 1
BoxLength = labda / Kn
N = p * BoxLength**3 / (kB * T)
NCell = (BoxLength / (0.5 * labda))**3
ppCell = (N / NCell)
cmList = []

'''
print('labda:', labda)
print('L:', BoxLength)
print('Cell size:', 0.7 * labda)
print('dt:', labda/math.sqrt(2 * R * T))
print('N:', N)
print('Ncells:', NCell)
print('ppCell:', ppCell)
'''


dt = 5e-11
TimeSteps = 40
NumberOfSimulations = 100
CubertNumberOfCells = 2
NumberOfCells = CubertNumberOfCells**3
NumberOfParticles = 336
HalfNumberOfParticles = 168
Ne = 1 #number of molecules per particle
Vc = BoxLength**3 / NumberOfCells  #Volume of cell
mass = [3.35e-27, 4.65e-26, 5.31e-26]
#mu = 0.5 * mass #reduced mass
NumberofProperties = 9
Particles = np.zeros((NumberOfParticles, NumberofProperties))
StartTemperature = np.zeros((NumberOfSimulations*NumberOfParticles,))
EndTemperature = np.zeros((NumberOfSimulations*NumberOfParticles,))
StartPosition = np.zeros((NumberOfSimulations*NumberOfParticles,))
EndPosition = np.zeros((NumberOfSimulations*NumberOfParticles,))
ParticleDistribution1 = np.zeros((TimeSteps+1,NumberOfSimulations))
ParticleDistribution2 = np.zeros((TimeSteps+1,NumberOfSimulations))
TemperatturePopulation1 = np.zeros((TimeSteps+1,))
TemperatturePopulation2 = np.zeros((TimeSteps+1,))
StartDistribution = 0.6
NumberOfParticlesPopulation1 = round(StartDistribution*NumberOfParticles)
NumberOfParticlesPopulation2 = NumberOfParticles - NumberOfParticlesPopulation1
Time= np.arange(TimeSteps+1)*dt

#seed = 3141592654
#key = jax.random.PRNGKey(seed)
rng = np.random.default_rng()

def generateParticles(Part):
    for i in range(NumberOfParticles):
        Part[i,0] = 1
        for j in range(1,4):
            Part[i,j] = rng.random()*BoxLength
        for k in range(4,7):
            Part[i,k] = random.choice([-1,1]) * np.sqrt(0.5 * 2* np.random.chisquare(1) * kB * T1 /  mass[1])
        Part[i,7] = 0.5 * mass[int(Part[i,0])] * (Part[i,4]**2 + Part[i,5]**2 + Part[i,6]**2)
        Part[i,8] = 0.5  * np.random.chisquare(2)  * kB * T2
    return Part

def generateDifferentDensities(Part):
    for i in range(NumberOfParticlesPopulation1):
        Part[i,0] = 0
        Part[i,1] = rng.random()*BoxLength*0.5
        for j in range(2,4):
            Part[i,j] = rng.random()*BoxLength
        for k in range(4,7):
            Part[i,k] = random.choice([-1,1]) * np.sqrt(0.5 * 2* np.random.chisquare(1) * kB * T /  mass[1])
        Part[i,7] = 0.5 * mass[1] * (Part[i,4]**2 + Part[i,5]**2 + Part[i,6]**2)
        Part[i,8] = 0.5 * 2 * np.random.chisquare(2)  * kB * T
    for i in range(NumberOfParticlesPopulation2):
        Part[i+NumberOfParticlesPopulation1,0] = 0
        Part[i+NumberOfParticlesPopulation1,1] = BoxLength-rng.random()*BoxLength*0.5
        for j in range(2,4):
            Part[i+NumberOfParticlesPopulation1,j] = rng.random()*BoxLength
        for k in range(4,7):
            Part[i+NumberOfParticlesPopulation1,k] = random.choice([-1,1]) * np.sqrt(0.5 * 2* np.random.chisquare(1) * kB * T /  mass[1])
        Part[i+NumberOfParticlesPopulation1,7] = 0.5 * mass[1] * (Part[i+NumberOfParticlesPopulation1,4]**2 + Part[i+NumberOfParticlesPopulation1,5]**2 + Part[i+NumberOfParticlesPopulation1,6]**2)
        Part[i+NumberOfParticlesPopulation1,8] = 0.5 * 2 * np.random.chisquare(2)  * kB * T
    return Part

def generateDifferentTemperatures(Part):
    for i in range(HalfNumberOfParticles):
        Part[i,0] = 0
        Part[i,1] = rng.random()*BoxLength*0.5
        for j in range(2,4):
            Part[i,j] = rng.random()*BoxLength
        for k in range(4,7):
            Part[i,k] = random.choice([-1,1]) * np.sqrt(0.5 * 2* np.random.chisquare(1) * kB * T1 /  mass[1])
        Part[i,7] = 0.5 * mass[1] * (Part[i,4]**2 + Part[i,5]**2 + Part[i,6]**2)
        #Part[i,7] = 0.5 * mass * (Part[i,4]**2) / kB
        Part[i,8] = 0.5 * 2 * np.random.chisquare(2)  * kB * T1
    for i in range(HalfNumberOfParticles):
        Part[i+HalfNumberOfParticles,0] = 0
        Part[i+HalfNumberOfParticles,1] = BoxLength-rng.random()*BoxLength*0.5
        for j in range(2,4):
            Part[i+HalfNumberOfParticles,j] = rng.random()*BoxLength
        for k in range(4,7):
            Part[i+HalfNumberOfParticles,k] = random.choice([-1,1]) * np.sqrt(0.5 * 2* np.random.chisquare(1) * kB * T2 /  mass[1])
        Part[i+HalfNumberOfParticles,7] = 0.5 * mass[1] * (Part[i+HalfNumberOfParticles,4]**2 + Part[i+HalfNumberOfParticles,5]**2 + Part[i+HalfNumberOfParticles,6]**2)
        #Part[i,7] = 0.5 * mass * (Part[i,4]**2) / kB
        Part[i+HalfNumberOfParticles,8] = 0.5 * 2 * np.random.chisquare(2)  * kB * T2
    return Part

def generateDifferentMolecules(Part):
    for i in range(NumberOfParticlesPopulation1):
        Part[i,0] = 1
        #Part[i,1] = rng.random()*BoxLength*0.5
        Part[i,1] = rng.random()*BoxLength
        for j in range(2,4):
            Part[i,j] = rng.random()*BoxLength
        for k in range(4,7):
            Part[i,k] = random.choice([-1,1]) * np.sqrt(0.5 * 2* np.random.chisquare(1) * kB * T1 /  mass[1])
        Part[i,7] = 0.5 * mass[1] * (Part[i,4]**2 + Part[i,5]**2 + Part[i,6]**2)
        #Part[i,7] = 0.5 * mass * (Part[i,4]**2) / kB
        Part[i,8] = 0.5 * 2 * np.random.chisquare(2)  * kB * T
    for i in range(NumberOfParticlesPopulation2):
        Part[i+NumberOfParticlesPopulation1,0] = 2
        #Part[i+NumberOfParticlesPopulation1,1] = BoxLength-rng.random()*BoxLength*0.5
        Part[i+NumberOfParticlesPopulation1,1] = rng.random()*BoxLength
        for j in range(2,4):
            Part[i+NumberOfParticlesPopulation1,j] = rng.random()*BoxLength
        for k in range(4,7):
            Part[i+NumberOfParticlesPopulation1,k] = random.choice([-1,1]) * np.sqrt(0.5 * 2* np.random.chisquare(1) * kB * T2 /  mass[2])
        Part[i+NumberOfParticlesPopulation1,7] = 0.5 * mass[2] * (Part[i+NumberOfParticlesPopulation1,4]**2 + Part[i+NumberOfParticlesPopulation1,5]**2 + Part[i+NumberOfParticlesPopulation1,6]**2)
        #Part[i,7] = 0.5 * mass * (Part[i,4]**2) / kB
        Part[i+NumberOfParticlesPopulation1,8] = 0.5 * 2 * np.random.chisquare(2)  * kB * T
    return Part

def updateCells(Part):
    for i in range(NumberOfCells):
        Cells.append([])
    
    for i in range(NumberOfParticles):
        c1 = math.floor((Part[i][1])*CubertNumberOfCells/BoxLength)
        c2 = math.floor((Part[i][2])*CubertNumberOfCells/BoxLength)
        c3 = math.floor((Part[i][3])*CubertNumberOfCells/BoxLength)
        CellNumber = c1 + CubertNumberOfCells * c2 + CubertNumberOfCells**2 * c3
        #can get list index out of range error

        Cells[CellNumber].append(i)

def acceptCollision(PartA, PartB, vrmax):
    vx = Particles[PartA, 4] - Particles[PartB, 4]
    vy = Particles[PartA, 5] - Particles[PartB, 5]
    vz = Particles[PartA, 6] - Particles[PartB, 6]
    v = math.sqrt(vx**2 + vy**2 + vz**2)
    if v /vrmax > rng.random():
        return True, v
    else:
        return False, v
    
def updateVelocities(PartA, PartB,vr):
    vcm1 = 0.5*(Particles[PartA, 4] + Particles[PartB, 4])
    vcm2 = 0.5*(Particles[PartA, 5] + Particles[PartB, 5])
    vcm3 = 0.5*(Particles[PartA, 6] + Particles[PartB, 6])
    q = 2 * rng.random() - 1
    phi = 2 * math.pi *rng.random()
    vr1 = vr * math.sqrt(1-q**2) * math.cos(phi)
    vr2 = vr * math.sqrt(1-q**2) *math.sin(phi)
    vr3 = vr * q
    Particles[PartA, 4] = vcm1 + 0.5*vr1
    Particles[PartA, 5] = vcm2 + 0.5*vr2
    Particles[PartA, 6] = vcm3 + 0.5*vr3
    Particles[PartB, 4] = vcm1 - 0.5*vr1
    Particles[PartB, 5] = vcm2 - 0.5*vr2
    Particles[PartB, 6] = vcm3 - 0.5*vr3

    Particles[PartA, 7] = 0.5 * mass[int(Particles[PartA,0])] * (Particles[PartA,4]**2 +Particles[PartA,5]**2 + Particles[PartA,6]**2)
    Particles[PartB, 7] = 0.5 * mass[int(Particles[PartB,0])] * (Particles[PartB,4]**2 +Particles[PartB,5]**2 + Particles[PartB,6]**2)

def calculateNofCollisions(Cell):
    vrmax = 0
    Nc = len(Cell)
    #print(Nc)
    for i in range(len(Cell)-1):
        for j in range(i+1, len(Cell)):
            v2 = (Particles[Cell[i], 4] - Particles[Cell[j], 4])**2 + (Particles[Cell[i], 5] - Particles[Cell[j], 5])**2 + (Particles[Cell[i], 6] - Particles[Cell[j], 6])**2
            if v2 > vrmax**2:
                vrmax = math.sqrt(v2)
    #print(vrmax)
    condition = Nc**2 * math.pi * d[1]*d[1] * vrmax * Ne * dt /(2 * Vc)
    #print(condition)
    return round(condition), vrmax

def collide():
    for i in range(NumberOfCells):
        NumberOfCollisions, vrmax = calculateNofCollisions(Cells[i])
        #print(NumberOfCollisions)
        #print(vrmax)
        for j in range(NumberOfCollisions):
            A = random.choice(Cells[i])
            B = random.choice(list(set(Cells[i])-set([A])))
            accepted, vr = acceptCollision(A,B,vrmax)
            if accepted:
                #vcm1 = 0.5*(Particles[A,4] + Particles[B,4])
                #vcm2 = 0.5*(Particles[A,5] + Particles[B,5])
                #vcm3 = 0.5*(Particles[A,6] + Particles[B,6])
                #updateVelocitiesBL(A, B, vr, vcm1, vcm2, vcm3)
                updateVelocities(A, B, vr)

def updatePositions(Part):
    for i in range(NumberOfParticles):
        #Part[i, 1] = (Part[i,1] + Part[i,4]*dt) % BoxLength
        if (Part[i,1] + Part[i,4]*dt) % (2 *BoxLength) > BoxLength:
            Part[i, 1] = BoxLength - ((Part[i,1] + Part[i,4]*dt) % BoxLength) #Spectral wall
        else:
            Part[i, 1] = (Part[i,1] + Part[i,4]*dt) % BoxLength
        Part[i, 2] = (Part[i,2] + Part[i,5]*dt) % BoxLength
        Part[i, 3] = (Part[i,3] + Part[i,6]*dt) % BoxLength

def averagePosition(Part):
    print(np.average(Part[:,1])/BoxLength)
    print(np.average(Part[:,2])/BoxLength)
    print(np.average(Part[:,3])/BoxLength)

def averageVelocity(Part):
    print(np.average(Part[:,4]))
    print(np.average(Part[:,5]))
    print(np.average(Part[:,6]))

def etrRatio(e):
    accepted = False

    while accepted == False:
        fracetr = rng.random()
        Petr = 4*fracetr*(1-fracetr)
        if Petr >= rng.random():
            accepted = True
    return e*fracetr

def updateVelocitiesBL(PartA, PartB, vr, vcm1, vcm2, vcm3):
    mu =mass[int(Particles[PartA,0])]*mass[int(Particles[PartB,0])]/(mass[int(Particles[PartA,0])]+mass[int(Particles[PartB,0])])
    #print(mu)
    e = 0.5 * mu * vr**2 + Particles[PartA,8] + Particles[PartB,8]
    cmList.append(e)
    etr = etrRatio(e)
    erot = e - etr
    erotA = erot * rng.random()
    Particles[PartA, 8] = erotA
    Particles[PartB, 8] = erot - erotA 

    vr = math.sqrt(2 * etr / mu)
    q = 1 - 2*rng.random()
    eps = 2 * math.pi * rng.random()
    vr1 = q * vr
    vr2 = math.sqrt(1-q**2) * math.cos(eps) * vr
    vr3 = math.sqrt(1-q**2) * math.sin(eps) * vr

    Particles[PartA, 4] = vcm1 - 0.5*vr1
    Particles[PartA, 5] = vcm2 - 0.5*vr2
    Particles[PartA, 6] = vcm3 - 0.5*vr3
    Particles[PartB, 4] = vcm1 + 0.5*vr1
    Particles[PartB, 5] = vcm2 + 0.5*vr2
    Particles[PartB, 6] = vcm3 + 0.5*vr3

    Particles[PartA, 7] = 0.5 * mass[int(Particles[PartA,0])] * (Particles[PartA,4]**2 +Particles[PartA,5]**2 + Particles[PartA,6]**2)
    Particles[PartB, 7] = 0.5 * mass[int(Particles[PartB,0])] * (Particles[PartB,4]**2 +Particles[PartB,5]**2 + Particles[PartB,6]**2)
    e = 0.5 * mu * vr**2 + Particles[PartA,8] + Particles[PartB,8]
    cmList.append(e)

def temperaturePlot():
    plt.hist(Particles[:,7]*2/3/kB)
    plt.show
      




# Run
#AllKeys = jax.random.split(key, NumberOfParticles*NumberofProperties).reshape(NumberOfParticles, NumberofProperties,2)
StartTemperatureEtr = np.zeros((NumberOfSimulations*NumberOfParticles,))
EndTemperatureEtr = np.zeros((NumberOfSimulations*NumberOfParticles,))
StartTemperatureErot = np.zeros((NumberOfSimulations*NumberOfParticles,))
EndTemperatureErot = np.zeros((NumberOfSimulations*NumberOfParticles,))

start = timeit.default_timer()

'''
for n in range(NumberOfSimulations):
    Particles = generateParticles(Particles)
    StartTemperatureEtr[n*NumberOfParticles:(n+1)*NumberOfParticles] = Particles[:,7]
    StartTemperatureErot[n*NumberOfParticles:(n+1)*NumberOfParticles] = Particles[:,8] 
    #StartPosition[n*NumberOfParticles:(n+1)*NumberOfParticles] = Particles[:,4]


    for t in range(TimeSteps):
        Cells = []
        ListPopulation1 = []
        ListPopulation2 = []
        updateCells(Particles)
        for i in range(len(Particles)):
            ListPopulation1.append(Particles[i, 7])
            ListPopulation2.append(Particles[i, 8])
        TemperatturePopulation1[t] = statistics.fmean(ListPopulation1)*2/3/kB
        TemperatturePopulation2[t] = statistics.fmean(ListPopulation2)*2/2/kB
        collide()
        updatePositions(Particles)

    Cells=[]
    updateCells(Particles)
    ListPopulation1 = []
    ListPopulation2 = []
    for i in range(len(Particles)):
        ListPopulation1.append(Particles[i, 7])
        ListPopulation2.append(Particles[i, 8])
    TemperatturePopulation1[TimeSteps] = statistics.fmean(ListPopulation1)*2/3/kB
    TemperatturePopulation2[TimeSteps] = statistics.fmean(ListPopulation2)*2/2/kB
    EndTemperatureEtr[n*NumberOfParticles:(n+1)*NumberOfParticles] = Particles[:,7]
    EndTemperatureErot[n*NumberOfParticles:(n+1)*NumberOfParticles] = Particles[:,8]
'''
    



for n in range(NumberOfSimulations):
    Particles = generateDifferentDensities(Particles)
    #StartTemperature[n*NumberOfParticles:(n+1)*NumberOfParticles] = Particles[:,7] 
    #StartPosition[n*NumberOfParticles:(n+1)*NumberOfParticles] = Particles[:,4]


    for t in range(TimeSteps):
        Cells = []
        ListPopulation1 = []
        ListPopulation2 = []
        updateCells(Particles)
        for i in range(NumberOfCells):
            for j in range(len(Cells[i])):
                if i%2 == 0:
                    ListPopulation1.append(Particles[Cells[i][j], 7])
                    
                else:
                    ListPopulation2.append(Particles[Cells[i][j], 7])
        #ParticleDistribution1[t][n] = statistics.fmean(ListPopulation1)-1
        #ParticleDistribution2[t][n] = statistics.fmean(ListPopulation2)-1
        ParticleDistribution1[t][n] = len(ListPopulation1)/NumberOfParticles
        ParticleDistribution2[t][n] = len(ListPopulation2)/NumberOfParticles
        #TemperatturePopulation1[t] = statistics.fmean(ListPopulation1)*2/3/kB
        #TemperatturePopulation2[t] = statistics.fmean(ListPopulation2)*2/3/kB
        collide()
        updatePositions(Particles)

    Cells=[]
    updateCells(Particles)
    ListPopulation1 = []
    ListPopulation2 = []
    for i in range(NumberOfCells):
        for j in range(len(Cells[i])):
            if i%2 == 0:
                ListPopulation1.append(Particles[Cells[i][j], 7])
            else:
                ListPopulation2.append(Particles[Cells[i][j], 7])
    #ParticleDistribution1[TimeSteps][n] = statistics.fmean(ListPopulation1)-1
    #ParticleDistribution2[TimeSteps][n] = statistics.fmean(ListPopulation2)-1
    ParticleDistribution1[TimeSteps][n] = len(ListPopulation1)/NumberOfParticles
    ParticleDistribution2[TimeSteps][n] = len(ListPopulation2)/NumberOfParticles
    #TemperatturePopulation1[TimeSteps] = statistics.fmean(ListPopulation1)*2/3/kB
    #TemperatturePopulation2[TimeSteps] = statistics.fmean(ListPopulation2)*2/3/kB
    #EndTemperature[n*NumberOfParticles:(n+1)*NumberOfParticles] = Particles[:,7]
    #EndPosition[n*NumberOfParticles:(n+1)*NumberOfParticles] = Particles[:,4]


pop1= np.zeros(TimeSteps+1,)
pop2= np.zeros(TimeSteps+1,)
for i in range(TimeSteps+1):
    pop1[i]= statistics.fmean(ParticleDistribution1[i])
    pop2[i]= statistics.fmean(ParticleDistribution2[i])

    
stop = timeit.default_timer()
print('Time: ', stop - start)


'''
plt.hist(StartTemperature*2/3/kB, bins=50, range=(0, 2000), alpha=0.5, label='Start')
plt.hist(EndTemperature*2/3/kB, bins=50, range=(0, 2000), alpha=0.5, label='End')
'''

'''
plt.hist(StartPosition, bins=20, range =(-4000,4000), alpha=0.5, label='Start')
plt.hist(EndPosition, bins=20, range =(-4000,4000), alpha=0.5, label='End')
'''
'''
plt.figure(1)
plt.scatter(Time, TemperatturePopulation1, label='Population1')
plt.scatter(Time, TemperatturePopulation2, label='Population2')
plt.ylim(0,12)
'''



plt.scatter(Time, pop1, alpha=0.5, label='Population1')
plt.scatter(Time, pop2, alpha=0.5, label='Population2')
plt.ylim(0,1)
plt.title('Particle distribution')

'''
fig,ax = plt.subplots(2)

ax[0].hist(StartTemperatureEtr*2/3/kB, bins=50, range=(0, 50), alpha=0.5, label='Etr')
ax[0].hist(StartTemperatureErot*2/2/kB, bins=50, range=(0, 50), alpha=0.5, label='Erot')
ax[0].set_title('Start temperature')

ax[1].hist(EndTemperatureEtr*2/3/kB, bins=50, range=(0, 50), alpha=0.5, label='Etr')
ax[1].hist(EndTemperatureErot*2/2/kB, bins=50, range=(0, 50), alpha=0.5, label='Erot')
ax[1].set_title('End temperature')
'''




plt.legend(loc='upper right')
plt.show()







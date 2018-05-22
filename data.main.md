
temperature 1.0

#VerletNVT dissipation rate ( viscosity)
gamma 0.1

#Number of fluid particles
numberFluidParticles  70000

boxSize 40 40 55

#Height of the top wall, from 0 to Lz
topWall_z 50


#Cut off between wall-fluid LJ interaction
cutOffWall_fluid    2.5

#wall-wall and fluid-fluid cut off
cutOff           1.12246204830937



#Files to read wall 
wallCoordinatesFile  wall.init
#p-p bonds
wallBondsFile        wall.bonds
#Bonds fixing to a certain height
wallBondsFileFP      wall.bondsFP

#Mass of a wall particle
wallParticleMass       100

#External force on the wall = A*sin(2*pi*w*t)
#A
wallOscilationAmplitude 5000
#w
wallOscilationWaveNumber 0.0005

#LJ epsilon between wall-wall and wall-fluid pairs
epsilonWall 50
#LJ epsilon between fluid-fluid pairs
epsilonFluid 1.2

#During the relaxation the wall does not oscilate
relaxSteps 1000

numberSteps 20000000

printSteps 5000

dt 0.005

outputName test

#Number of bins for the velocity profile histogram
nbinsHisto 50

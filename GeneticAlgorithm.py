# The Electric Vehicle Multi Trip Pickup and Delivery Problem with Time
# Windows and Transfers: an Approach using a Hybrid Genetic Algorithm

# D. Hiemstra and W. Goossens

# First, we import relevant libraries and import the instance dataset.
import heapq
import numpy as np
from scipy import linalg as LA
from scipy.sparse import csgraph
import pandas as pd
import random
from random import randint
from copy import copy, deepcopy
import sys
import os
import csv
import gc
import time
MAXVALUE = sys.float_info.max

# Read .csv data (only the first instance)
jobid = os.environ["SLURM_ARRAY_TASK_ID"]
skipNRows = int(jobid)
rowsToBeSkipped = [0]
for i in range(0, skipNRows):
    rowsToBeSkipped.append(i)
instance = pd.read_csv("./oracs/instances_v1.csv", header = None, nrows = 1, skiprows = rowsToBeSkipped)

instance = instance.values.tolist()
instance = instance[0]

# Instance variables
instanceSettings = int(instance[0])
instance.pop(0)
replicationNumber = int(instance[0])
instance.pop(0)

idx = int(instance[0])
n = int(instance[1])
timeHorizon = int(instance[2])
kVehicles = int(instance[3])
qCapacity = int(instance[4])
dBattery = int(instance[5])
eEnergy = float(instance[6])
gRecharging = float(instance[7])
tTime = float(instance[8])

# Locations and their deadlines
x0 = instance[9]
y0 = instance [10]

requests = np.zeros((2 * n, 2))

for i in range(0, 2 * n):
    requests[i, 0] = instance[11 + 2 * i]
    requests[i, 1] = instance[11 + 2 * i + 1]

deadlines = np.zeros(n+1)

for i in range(1, n + 1):
    deadlines[i] = instance[10 + 4 * n + i]


# Using the data on all locations, we can create a distance matrix.
distanceMatrix = np.zeros((2 * n + 1, 2 * n + 1))

# Distance from the depot to each location and vice versa
for i in range(1, 2 * n + 1):
    distanceMatrix[0, i] = np.sqrt((requests[i - 1, 0] - x0) ** 2 + (requests[i - 1, 1] - y0) ** 2)
    distanceMatrix[i, 0] = np.sqrt((requests[i - 1, 0] - x0) ** 2 + (requests[i - 1, 1] - y0) ** 2)

# Distance from each location to every other location and vice versa
for i in range(0, 2 * n):
    for j in range(0, 2 * n):
        xlen = ((requests[i, 0] - requests[j, 0]) ** 2)
        ylen = ((requests[i, 1] - requests[j, 1]) ** 2)
        
        distanceMatrix[i + 1, j + 1] = np.sqrt(xlen + ylen)

# First, we are going to create the Chromosome class.
def tourdistance(locations, arrivalTimes):
    nLocations = len(locations)
    
    if nLocations == 0:
        return(0)
    
    delta = distanceMatrix[0, locations[0]]
    arrivalTimes[0] = delta * tTime
    
    for i in range(1, nLocations):
        delta = delta + distanceMatrix[locations[i - 1], locations[i]]
        arrivalTimes[i] = delta * tTime
    
    delta = delta + distanceMatrix[locations[nLocations - 1], 0]
    
    arrivalTimes[nLocations] = delta * tTime
    
    return(delta)


# The chromosome class describes a single tour.
class Chromosome:
    # This function initiates an empty Chromsome object
    def __init__(self):
        self.nLocations = 0
        self.pickups = []
        self.deliveries = []
        self.notReachable = []
        self.tour = []
        self.Delta = 0
        self.arrivalTimes = []
    
    # The following function adds a number of locations to the chromosome. It does not check for feasibility.
    def addSequence(self, sequence):
        self.nLocations = len(sequence)
        
        if self.nLocations > 0:
            pickups = []
            deliveries = []
            
            for i in range(0, self.nLocations):
                if sequence[i] <= n:
                    pickups.append(sequence[i])
                else:
                    deliveries.append(sequence[i])
            
            self.pickups = list(pickups)
            self.deliveries = list(deliveries)
            
            self.tour = tourGeneration(pickups, deliveries, self.notReachable)
            
            self.arrivalTimes = np.zeros((self.nLocations + 1))
            self.Delta = tourdistance(self.tour, self.arrivalTimes)
    
    def tourChecker(self, tour):
        noPickups = [x for x in tour if x > n and (x - n) not in tour]
        currentLoad = len(noPickups)
        
        for i in range(0, len(tour)):
            if tour[i] > n and (tour[i] - n) in tour[i:]:
                return(False)
            
            if tour[i] > n:
                currentLoad -= 1
            else:
                currentLoad += 1
                if currentLoad > qCapacity:
                    return(False)
        return(True)
    
    # This functions removes a specific location from the current tour.
    def removeLocation(self, location):
        if location in self.tour:
            tour = list(self.tour)
            
            tour.remove(location)
            
            if not self.tourChecker(tour):
                return(False)
            else:
                if location <= n:
                    self.pickups.remove(location)
                else:
                    self.deliveries.remove(location)

                self.tour.remove(location)
                self.nLocations -= 1
                self.Delta = tourdistance(self.tour, self.arrivalTimes)
                return(True)
        else:
            return(False)
    
    # The following function removes the location at a specific position of the tour.
    def removeLocationAtPosition(self, position):
        if position <= self.nLocations:
            location = self.tour[position]
            tour = list(self.tour)
            tour.remove(location)
            
            if not self.tourChecker(tour):
                return(False)
            else:
                if location <= n:
                    self.pickups.remove(location)
                else:
                    self.deliveries.remove(location)

                self.tour.remove(location)
                self.nLocations -= 1
                self.Delta = tourdistance(self.tour, self.arrivalTimes)
                return(True)
        else:
            return(False)
    
    
    # This function removes a specific location from the current tour and recalculates the nearest-neighbour tour.
    def removeLocationNewTour(self, location):   
        if location in self.tour:
            pickups = list(self.pickups)
            deliveries = list(self.deliveries)
            
            if location <= n:
                pickups.remove(location)
            else:
                deliveries.remove(location)
            
            notReachable = []
            tour = tourGeneration(pickups, deliveries, notReachable)
            
            if notReachable == [] and self.tourChecker(tour):
                if location <= n:
                    self.pickups.remove(location)
                else:
                    self.deliveries.remove(location)
                
                self.tour = tour
                self.nLocations -= 1
                self.arrivalTimes = np.zeros((self.nLocations + 1))
                self.Delta = tourdistance(self.tour, self.arrivalTimes)
                return(True)
            else:
                return(False)
        else:
            return(False)
    
    
    # This function adds a location to the current tour and recalculates the nearest-neighbour tour.
    def addLocation(self, location):
        if location not in self.tour:
            pickups = list(self.pickups)
            deliveries = list(self.deliveries)

            if location <= n:
                pickups.append(location)
            else:
                deliveries.append(location)

            notReachable = []
            tour = tourGeneration(pickups, deliveries, notReachable)

            if notReachable == [] and self.tourChecker(tour):
                if location <= n:
                    self.pickups.append(location)
                else:
                    self.deliveries.append(location)

                self.tour = tour
                self.nLocations += 1
                self.arrivalTimes = np.zeros((self.nLocations + 1))
                self.Delta = tourdistance(self.tour, self.arrivalTimes)
                return(True)
            else:
                return(False)
        else:
            return(False)
    
    def addMissingLocation(self, location):
        pickups = list(self.pickups)
        deliveries = list(self.deliveries)
        
        if location <= n:
            pickups.append(location)
        else:
            deliveries.append(location)
        
        notReachable = []
        tour = tourGeneration(pickups, deliveries, notReachable)
        
        if notReachable == [] and self.tourChecker(tour):
            if location <= n:
                self.pickups.append(location)
            else:
                self.deliveries.append(location)
            
            self.tour = tour
            self.nLocations += 1
            self.arrivalTimes = np.zeros((self.nLocations + 1))
            self.Delta = tourdistance(self.tour, self.arrivalTimes)
            return(True)
        else:
            return(False)
        
    # This function adds a location at a specific position in the tour, if possible. If not, it returns False.
    def addLocationAtPosition(self, location, position):
        pickups = list(self.pickups)
        deliveries = list(self.deliveries)
        tour = list(self.tour)
        nLocations = self.nLocations + 1
        arrivalTimes = np.zeros((nLocations + 1))
        
        if (position <= self.nLocations):
            tour.insert(position, location)
        else:
            tour.append(location)
        
        
        if ((tourdistance(tour, arrivalTimes) * eEnergy) > dBattery):
            return(False)
        
        if location <= n:
            if (location + n in tour[0:(position - 1)]):
                return(False)
            
            pickups.append(location)
        else:
            if (location - n in tour[(position + 1):nLocations]):
                return(False)
            
            deliveries.append(location)
                
        if not self.tourChecker(tour):
            return(False)
        
        # If the weight and driving range constraints are not violated, we can add the location to the chromosome.
        self.pickups = list(pickups)
        self.deliveries = list(deliveries)
        self.tour = list(tour)
        self.arrivalTimes = list(arrivalTimes)
        self.nLocations += 1
        self.Delta = tourdistance(self.tour, self.arrivalTimes)
        
        return(True)


# This function creates a tour based on the nearest-possible-neighbour. It returns the tour and appends the
# notReachable argument with unreachable locations.
def tourGeneration(pickups, deliveries, notReachable):
    if not pickups and not deliveries:
        return []
    current = 0
    tour = []
    tourLength = 0
    possibleDeliveries = []
    weight = 0
    
    # First, it calculates the weight of the vehicle at the start of the tour. If the weight is exceeded, the
    # last deliveries are not reachable.
    for i in range(0, len(deliveries)):
        if((deliveries[i] - n) not in pickups):
            if weight < qCapacity:
                weight += 1
                possibleDeliveries.append(deliveries[i])
            else:
                notReachable.append(deliveries[i])
    
    while (len(pickups) > 0) | (len(possibleDeliveries) > 0):
        # If there are pickups available and there is room for extra cargo, we enter this statement. We check
        # the closest location to our current location, either a pickup or a delivery.
        impossiblePickups = []
        impossibleDeliveries = []
        if (len(pickups) > 0) & (weight < qCapacity):
            nextVisit = 0
            shortest = distanceMatrix[current, pickups[nextVisit]]
            delivery = False
            
            if ((tourLength + shortest + distanceMatrix[pickups[nextVisit], 0]) * eEnergy) > dBattery:
                notReachable.append(pickups[nextVisit])
                impossiblePickups.append(0)
            
            for i in range(1, len(pickups)):
                addedDistance = distanceMatrix[current, pickups[i]]
                if ((tourLength + addedDistance + distanceMatrix[pickups[i], 0]) * eEnergy) > dBattery:
                    notReachable.append(pickups[i])
                    impossiblePickups.append(i)
                elif addedDistance < shortest:
                    nextVisit = i
                    shortest = addedDistance
            
            for i in range(0, len(possibleDeliveries)):
                k = i
                if (k + 1) > len(possibleDeliveries):
                    k = k - 1
                
                addedDistance = distanceMatrix[current, possibleDeliveries[k]]
                
                if ((tourLength + addedDistance + distanceMatrix[possibleDeliveries[k], 0]) * eEnergy) > dBattery:
                    notReachable.append(possibleDeliveries[k])
                    impossibleDeliveries.append(k)
                elif addedDistance < shortest:
                    nextVisit = i
                    shortest = addedDistance
                    delivery = True
            
            # If there is a pickup location visited, we can visit its associated delivery location.
            if (delivery) & (nextVisit != -1):
                current = possibleDeliveries[nextVisit]
                tour.append(current)
                weight -= 1
                tourLength = tourLength + shortest
                del possibleDeliveries[nextVisit]
                
                impossiblePickups.sort()
                for i in range(0, len(impossiblePickups)):
                    del pickups[impossiblePickups[i] - i]
            elif (nextVisit != -1):
                current = pickups[nextVisit]
                tour.append(current)
                weight += 1
                tourLength = tourLength + shortest
                
                if nextVisit not in impossiblePickups:
                    impossiblePickups.append(nextVisit)
                
                impossiblePickups.sort()
                for i in range(0, len(impossiblePickups)):
                    del pickups[impossiblePickups[i] - i]
                
                
                for i in range(0, len(deliveries)):
                    if (current + n) == deliveries[i]:
                        possibleDeliveries.append(deliveries[i])
        # If there are no pickups possible, we check whether there are deliveries possible
        elif (len(possibleDeliveries) > 0):
            nextVisit = 0
            shortest = distanceMatrix[current, possibleDeliveries[nextVisit]]
            
            if ((tourLength + shortest + distanceMatrix[possibleDeliveries[nextVisit], 0]) * eEnergy) > dBattery:
                notReachable.append(possibleDeliveries[nextVisit])
                nextVisit = -1
                impossibleDeliveries.append(0)
            
            for i in range(1, len(possibleDeliveries)):
                addedDistance = distanceMatrix[current, possibleDeliveries[i]]
                if ((tourLength + addedDistance + distanceMatrix[possibleDeliveries[i], 0]) * eEnergy) > dBattery:
                    notReachable.append(possibleDeliveries[i])         
                    impossibleDeliveries.append(i)
                elif addedDistance < shortest:
                    nextVisit = i
                    shortest = addedDistance
            
            if nextVisit != -1:
                current = possibleDeliveries[nextVisit]
                tour.append(current)
                weight -= 1
                tourLength = tourLength + shortest
                impossibleDeliveries.append(nextVisit)
            
            impossibleDeliveries.sort()
            for i in range(0, len(impossibleDeliveries)):
                del possibleDeliveries[impossibleDeliveries[i] - i]
        
        # If no pickups are possible, no deliveries are possible and the weight is exceeded, we have some
        # unreachable locations.
        elif weight >= qCapacity:
            for i in range(0, len(pickups)):
                notReachable.append(pickups[i])
            for i in range(0, len(deliveries)):
                if deliveries[i] not in tour:
                    notReachable.append(deliveries[i])
            break
    return(tour)


# This function selects the next Chromosome to be scheduled, based on the nearest deadline of all lower dependent
# jobs and the total driving time up until then.
def selectNextJob(dependence, tightness, pos):
    upperDependence = [i for i, x in enumerate(dependence[pos, :]) if x == 1]
    if not upperDependence:
        return(pos)
    else:
        upperTightness = [(tightness[x], x) for x in upperDependence]
        
        tightestUpper = min(upperTightness)[1]
        return(selectNextJob(dependence, tightness, tightestUpper))

# This calculates the total driving time that is needed for the selectNextJob-function.
def totalDrivingtime(dependence, chromosomeSet, pos):
    drivingTime = chromosomeSet[pos].Delta
    test = [i for i,x in enumerate(dependence[pos,]) if x==1]
    
    if sum(test) == 0:
        return(drivingTime)
    else:
        upperDependence = test
        upperTimes = np.zeros(len(upperDependence))
        
        for i in range(0, len(upperDependence)):
            upperTimes[i] = totalDrivingtime(dependence, chromosomeSet, upperDependence[i])
        drivingTime += max(upperTimes)
        
        return(drivingTime)

# The individual class is used to store a solution
class Individual:
    # Initializing a Individual creates an empty object.
    def __init__(self):
        self.chromosomeSet = []
        self.sequence = []
        self.visits = [Chromosome() for x in range(0, 2 * n + 1)]
        self.visitsChild =[[Chromosome() for j in range(0,2)] for i in range(0,2 * n + 1)]
        self.goal = 0
        self.dependence = []
        
        self.jobsPerVehicle = np.zeros(kVehicles)
        self.arrivals = []
        self.infeasibility = np.zeros(2)
        self.lateDeadlines = [0 for x in range(0, n + 1)]
        self.missingPickups=[]
        self.missingDeliveries=[]
        self.duplicatesPickups=[]
        self.duplicatesDeliveries=[]
        
        self.vehicleTrips = [[] for x in range(0, kVehicles)]
        self.vehicleLoad = [[] for x in range(0, kVehicles)]
        self.vehicleDepartures = [[] for x in range(0, kVehicles)]
        self.vehicleTripLengths = [[] for x in range(0, kVehicles)]
        self.vehicleMaxLoad = [[] for x in range(0, kVehicles)]
    
    
    def __eq__(self, other):
        return(self.goal == other.goal)
    
    
    def __lt__(self, other):
        return(self.goal < other.goal)
    
    
    def updateIndividual(self):
        self.goal = 0
        self.visits = [Chromosome() for x in range(0, 2 * n + 1)]
        emptyChromosomes = []
        
        for chromosome in self.chromosomeSet:
            self.goal = self.goal + chromosome.Delta
            
            if not chromosome.tour:
                emptyChromosomes.append(self.chromosomeSet.index(chromosome))
            
            for location in chromosome.tour:
                self.visits[location] = chromosome
        
        emptyChromosomes.sort()
        for i in range(0, len(emptyChromosomes)):
            del self.chromosomeSet[emptyChromosomes[i] - i]
        
        self.updateDependenceForest()
        
        if loopChecker(self, []) == []:
            self.scheduler()
            self.calculateInfeasibility()
            return(True)
        else:
            return(False)
    
    
    # This function adds a Chromosome object to the Individual object.
    def addChromosome(self, newChromosome):
        chromosome = deepcopy(newChromosome)
        self.chromosomeSet.append(chromosome)
        
        for location in chromosome.tour:
            self.visits[location] = chromosome
        
        self.goal = self.goal + chromosome.Delta
    
    
   #Functions for Child
    def addChromosomeChild(self, newChromosome):
        chromosome = deepcopy(newChromosome)
        self.chromosomeSet.append(chromosome)
        
        for location in chromosome.tour:
            if self.visitsChild[location][0].nLocations == 0:
                self.visitsChild[location][0] = chromosome
                self.visits[location] = chromosome
            else:
                self.visitsChild[location][1] = chromosome
            
            self.goal = self.goal + chromosome.Delta
    
    
    #AddingLocation
    def addMissingLocation(self, selectedInsertion, location):
        chromosome = self.chromosomeSet[selectedInsertion]
        chromosome.addMissingLocation(location)
        self.visitsChild[location][0] = chromosome
        
        if location <= n:
            self.missingPickups.remove(location)
        else:
            self.missingDeliveries.remove(location)
    
    
    #Removing Duplicates in the individual 
    def removeDuplicateLocation(self, index, location):
        chromosome = self.chromosomeSet[index]
        chromosome.removeLocationNewTour(location)
        
        
        self.updateDependenceForestChild()
        
        if self.visitsChild[location][0] == chromosome:
            self.visitsChild[location][0] = self.visitsChild[location][1] #replace the first column index with the second
            self.visitsChild[location][1] = Chromosome()
        elif self.visitsChild[location][1] == chromosome:
            self.visitsChild[location][1] = Chromosome()
        else:
            return
    
    def removeNonDuplicateLocation(self, chromosome, location):
        chromosome.removeLocationNewTour(location)
        
        if location <= n:
            self.missingPickups.append(location)
        else:
            self.missingDeliveries.append(location)
        
        self.visitsChild[location][0] = Chromosome()
        self.visitsChild[location][1] = Chromosome()
    
    
    def updateDependenceForestChild(self):
        self.missingPickups = []
        self.missingDeliveries = []
        self.duplicatesPickups = []
        self.duplicatesDeliveries = []
        self.goal = 0
        dependence = np.zeros((len(self.chromosomeSet), len(self.chromosomeSet)))
        chromosomeSet = self.chromosomeSet
        visitsChild = self.visitsChild
        
        for i in chromosomeSet:
            self.goal += i.Delta
        
        allLocations = []
        
        for chromosome in chromosomeSet:
            for i in range(0, len(chromosome.tour)):
                if chromosome.tour[i] in allLocations:
                    if i <= n:
                        self.duplicatesPickups.append(chromosome.tour[i])
                    else:
                        self.duplicatesDeliveries.append(chromosome.tour[i])
                    
                    location0 = self.visitsChild[i][0] == chromosome
                    location1 = self.visitsChild[i][1] == chromosome
                    
                    if not location0 and not location1:
                        if self.visitsChild[i][0].nLocations > 0:
                            self.visitsChild[i][1] = chromosome
                        else:
                            self.visitsChild[i][0] = chromosome
                else:
                    allLocations.append(chromosome.tour[i])
                    
                    location0 = self.visitsChild[i][0] == chromosome
                    location1 = self.visitsChild[i][1] == chromosome
                    
                    if not location0 and not location1:
                        if self.visitsChild[i][0].nLocations > 0:
                            self.visitsChild[i][1] = chromosome
                        else:
                            self.visitsChild[i][0] = chromosome
        
        for i in range(1, n + 1):
            if i not in allLocations:
                self.missingPickups.append(i)
        
        for i in range(n + 1, 2 * n + 1):
            if i not in allLocations:
                self.missingDeliveries.append(i)
        
        for i in range(1, n + 1):  
            if self.visitsChild[i][0].nLocations>0 and self.visitsChild[i+n][0].nLocations>0:
                dependence[chromosomeSet.index(self.visitsChild[i + n][0]), chromosomeSet.index(self.visitsChild[i][0])] = 1
            if (self.visitsChild[i][1].nLocations>0 and self.visitsChild[i+n][1].nLocations>0):
                dependence[chromosomeSet.index(self.visitsChild[i+n][0]), chromosomeSet.index(self.visitsChild[i][1])] = 1
                dependence[chromosomeSet.index(self.visitsChild[i+n][1]), chromosomeSet.index(self.visitsChild[i][1])] = 1
                dependence[chromosomeSet.index(self.visitsChild[i+n][1]), chromosomeSet.index(self.visitsChild[i][0])] = 1
            elif (self.visitsChild[i][1].nLocations > 0 and self.visitsChild[i+n][1].nLocations == 0 and self.visitsChild[i+n][0].nLocations>0):
                dependence[chromosomeSet.index(self.visitsChild[i+n][0]), chromosomeSet.index(self.visitsChild[i][1])] = 1
    
            elif self.visitsChild[i][1].nLocations==0 and self.visitsChild[i+n][1].nLocations>0 and self.visitsChild[i][0].nLocations>0:
                dependence[chromosomeSet.index(self.visitsChild[i+n][1]), chromosomeSet.index(self.visitsChild[i][0])] = 1     
        
        np.fill_diagonal(dependence, 0)
        self.dependence = dependence
    
    
    # The following function updates the dependence forest. It has to be called before scheduling the Individual.
    # dependence forest. It has to be called before scheduling the Individual.
    def updateDependenceForest(self):
        dependence = np.zeros((len(self.chromosomeSet), len(self.chromosomeSet)))
        chromosomeSet = self.chromosomeSet
        visits = self.visits
       
        for i in range(1, n + 1):
            if self.visits[i] != self.visits[i + n]:
                dependence[chromosomeSet.index(visits[i + n]), chromosomeSet.index(visits[i])] = 1
       
        self.dependence = dependence
    
        
    # This function prints the information of the Individual.
    def printer(self):
        print("Chromosome tours:")
        self.goal = 0
        for chrom in self.chromosomeSet:
            print(chrom.tour)
            self.goal = chrom.Delta + self.goal
        print("Dependence tree:")
        print(self.dependence)
        print(self.goal)

    def solutionPrinter(self):
        filename = 'oracs_' + str(idx) + '.csv'
        fileopener = open(filename, 'w', newline = '')
        
        with fileopener:
            filewriter = csv.writer(fileopener, delimiter = ',')
            filewriter.writerow(str(2))
            filewriter.writerow(str(idx))
            goalvalue = ["%.2f" % self.goal]
            filewriter.writerow(goalvalue)
            for i in range(0, kVehicles):
                filewriter.writerow([])
                filewriter.writerow([i, self.jobsPerVehicle[i]])
                filewriter.writerow(self.vehicleTrips[i])
                filewriter.writerow(self.vehicleLoad[i])
                filewriter.writerow(self.vehicleDepartures[i])
                filewriter.writerow(self.vehicleTripLengths[i])
                filewriter.writerow(self.vehicleMaxLoad[i])
            
            
    # This function prints the schedule of the Individual.
    def schedulePrinter(self):
        for i in range(0, kVehicles):
            print("Vehicle ", i, ":")
            print(self.jobsPerVehicle[i])
            print(self.assignedJobs[i])
            print(self.arrivals[i])
    
    
    # This function prints the solution in the right format
    
    # The following creates a schedule for an Individual.
    def scheduler(self):
        dependence = self.dependence.copy()
        chromosomeSet = self.chromosomeSet
        sequenceLength = len(chromosomeSet)
       
        jobsPerVehicle = [0 for x in range(0, kVehicles)]
        assignedJobs = [[None for x in range(0, sequenceLength + 1)] for i in range(0, kVehicles)]
        chromosomeAssignedToVehicle = [[-1 for x in range(0, sequenceLength)] for i in range(0, 2)]
        arrivals = np.zeros([kVehicles, sequenceLength + 1])
        recharged = np.zeros([kVehicles, sequenceLength + 1])
        rechargedVehicles = np.zeros(kVehicles)
        
        vehicleTrips = [[] for x in range(0, kVehicles)]
        vehicleLoad = [[] for x in range(0, kVehicles)]
        vehicleDepartures = [[] for x in range(0, kVehicles)]
        vehicleTripLengths = [[] for x in range(0, kVehicles)]
        vehicleMaxLoad = [[] for x in range(0, kVehicles)]
        
        earliestDeadline = np.zeros(sequenceLength)
        tightness = [0 for x in range(0, sequenceLength)]
        
        sequence = []
        
        for i in range(0, sequenceLength):
            earliest = timeHorizon
           
            for j in range(0, len(chromosomeSet[i].deliveries)):
                if deadlines[(chromosomeSet[i].deliveries[j] - n)] < earliest:
                    earliest = deadlines[(chromosomeSet[i].deliveries[j] - n)]
           
            earliestDeadline[i] = earliest
       
        for i in range(0, sequenceLength):
            tightness[i] = totalDrivingtime(dependence, chromosomeSet, i) - earliestDeadline[i]
        
        while len(sequence) < sequenceLength:
            tightestDeadline = tightness.index(min(tightness))
            nextJobIndex = selectNextJob(dependence, tightness, tightestDeadline)
            nextJob = chromosomeSet[nextJobIndex]
            sequence.append(nextJob)
            
            for i in range(0, kVehicles):
                rechargedVehicles[i] = recharged[i][jobsPerVehicle[i]]
            
            vehicle = rechargedVehicles.argmin()
            jobsPerVehicle[vehicle] += 1
            assignedJobs[vehicle][jobsPerVehicle[vehicle]] = nextJobIndex
            chromosomeAssignedToVehicle[0][nextJobIndex] = vehicle
            
            if sum(self.dependence[nextJobIndex, :]) == 0:
                startingTime = recharged[vehicle][jobsPerVehicle[vehicle] - 1]
            else:
                upperDependence = [i for i, x in enumerate(self.dependence[nextJobIndex, :]) if x == 1]
                waitForVehicles = [chromosomeAssignedToVehicle[1][x] for x in upperDependence]
                possibleStart = max(waitForVehicles)
                startingTime = max(possibleStart, recharged[vehicle][jobsPerVehicle[vehicle] - 1])
            
            arrivalTime = tTime * nextJob.Delta + startingTime
            chromosomeAssignedToVehicle[1][nextJobIndex] = arrivalTime
            arrivals[vehicle][jobsPerVehicle[vehicle]] = arrivalTime
            
            rechargedTime = arrivalTime + (nextJob.Delta * eEnergy) / gRecharging
            recharged[vehicle][jobsPerVehicle[vehicle]] = rechargedTime
            
            weight = 0
            
            for i in range(0, len(nextJob.tour)):
                if nextJob.tour[i] > n and nextJob.tour[i] - n not in nextJob.tour:
                    weight += 1
            
            maxWeight = weight
            
            vehicleTrips[vehicle].append(0)
            vehicleDepartures[vehicle].append(startingTime)
            vehicleLoad[vehicle].append(weight)
            
            for i in range(0, len(nextJob.tour)):
                vehicleTrips[vehicle].append(nextJob.tour[i])
                vehicleDepartures[vehicle].append(nextJob.arrivalTimes[i] + startingTime)
                
                if nextJob.tour[i] > n:
                    weight -= 1
                else:
                    weight += 1
                    
                    if weight > maxWeight:
                        maxWeight = weight
                
                vehicleLoad[vehicle].append(weight)
            
            vehicleTripLengths[vehicle].append(nextJob.Delta)
            vehicleMaxLoad[vehicle].append(maxWeight)
                
            
            tightness[nextJobIndex] = MAXVALUE
            
            dependence[:, nextJobIndex] = 0
        
        self.sequence = sequence
        self.jobsPerVehicle = jobsPerVehicle
        
        for i in range(0, kVehicles):
            if jobsPerVehicle[i] > 0:
                arrivalTime = vehicleDepartures[i][-1] + tTime * distanceMatrix[vehicleTrips[i][-1], 0]
                vehicleTrips[i].append(0)
                vehicleLoad[i].append(-1)
                vehicleDepartures[i].append(arrivalTime)
            else:
                vehicleTrips[i].append(0)
                vehicleDepartures[i].append(0)
                vehicleLoad[i].append(0)
                vehicleTripLengths[i].append(0)
                vehicleMaxLoad[i].append(0)
                
        
        self.vehicleTrips = vehicleTrips
        self.vehicleLoad = vehicleLoad
        self.vehicleDepartures = vehicleDepartures
        self.vehicleTripLengths = vehicleTripLengths
        self.vehicleMaxLoad = vehicleMaxLoad
    
    
    def calculateInfeasibility(self):
        infeasibility = self.infeasibility.copy()
        lateDeadlines = [0 for x in range(0, n + 1)]
        
        for i in range(1, n + 1):
            for j in range(0, len(self.vehicleTrips)):
                if (n + i) in self.vehicleTrips[j]:
                    k = self.vehicleTrips[j].index(n + i)
                    arrivalTime = self.vehicleDepartures[j][k]
                    if arrivalTime > deadlines[i]:
                        lateDeadlines[i] += arrivalTime - deadlines[i]
                    break
        
        infeasibility[0] = sum(lateDeadlines)
        self.lateDeadlines = lateDeadlines
        self.infeasibility = infeasibility

# Initialisation procedure H1
def InitializeH1():
    newIndividual = Individual()
    pickupSet = list(range(1, n + 1))
    deliverySet = list(range(1, n + 1))
    q = 1 / qCapacity
    
    while deliverySet != []:
        notFull = True
        newChromosome = Chromosome()
        
        while notFull:
            if pickupSet != []:
                newLocation = random.choice(pickupSet)
                notFull = newChromosome.addLocation(newLocation)
                
                if notFull:
                    pickupSet.remove(newLocation)
                    if (newLocation + n) in deliverySet:
                        notFull = newChromosome.addLocation(newLocation + n)
                        if notFull:
                            deliverySet.remove(newLocation)
            elif deliverySet != []:
                newLocation = random.choice(deliverySet)
                notFull = newChromosome.addLocation(newLocation + n)
                
                if notFull:
                    deliverySet.remove(newLocation)
            else:
                break
            
            if notFull:
                notFull = np.random.choice([True, False], p = [1 - q, q])
        
        newIndividual.addChromosome(newChromosome)
    
    newIndividual.updateIndividual()
    
    return(newIndividual)

# Intialisation procedure H2
def InitializeH2():
    newIndividual = Individual()
    pickupSet = [None] * (n)
    deliverySet = [None] * (n)
    q = 1 / qCapacity
    
    for i in range(0, n):
        pickupSet[i] = i + 1
        deliverySet[i] = i + 1
    
    toBePickedUp = []
    
    while deliverySet != []:
        notFull = True
        newChromosome = Chromosome()
        
        while notFull:
            anotherPickup = False
            
            if pickupSet != []:
                newLocation = random.choice(pickupSet)
                notFull = newChromosome.addLocation(newLocation)
                
                if notFull:
                    pickupSet.remove(newLocation)
                    toBePickedUp.append(newLocation)
                    anotherPickup = np.random.choice([True, False], p = [1 - q, q])
                    
                    if anotherPickup and pickupSet != []:
                        newLocation = random.choice(pickupSet)
                        notFull = newChromosome.addLocation(newLocation)
                        
                        if notFull:
                            pickupSet.remove(newLocation)
                            toBePickedUp.append(newLocation)
                    
                    elif deliverySet != []:
                        newLocation = random.choice(toBePickedUp)
                        notFull = newChromosome.addLocation(newLocation + n)
                        
                        if notFull:
                            toBePickedUp.remove(newLocation)
                            deliverySet.remove(newLocation)
            
            elif deliverySet != []:
                newLocation = random.choice(toBePickedUp)
                notFull = newChromosome.addLocation(newLocation + n)
                
                if notFull:
                    toBePickedUp.remove(newLocation)
                    deliverySet.remove(newLocation)
            else:
                notFull = False
            
            if notFull:
                notFull = np.random.choice([True, False], p = [1 - q, q])
        
        newIndividual.addChromosome(newChromosome)
    
    newIndividual.updateIndividual()
    
    return(newIndividual)

# This function checks for loops in the dependence forest. It makes use of Tarjan's algo.
def loopChecker(individual, potentialDependence):
    if potentialDependence == []:
        dependence = individual.dependence[0:len(individual.chromosomeSet), 0:len(individual.chromosomeSet)]
    else:               
        dependence = potentialDependence
    
    nChromosomes = len(dependence)
        
    # Use Tarjan's algorithm to find strongly connected components of our dependence forest
    loops = csgraph.connected_components(dependence, directed = True, connection = 'strong')
    
    toBeAltered = []
    
    if loops[0] < nChromosomes:
        loopGroups = []
        
        for i in range(0, loops[0]):
            loopGroups.append([j for j, x in enumerate(loops[1]) if x == i])
        
        for i in range(0, loops[0]):
            loopGroupLength = len(loopGroups[i])
            
            if loopGroupLength > 1:
                maxDifference = []
                upperMinusLower = [None for x in range(0, loopGroupLength)]
                
                for j in range(0, len(loopGroups[i])):
                    upperDependent = 0
                    lowerDependent = 0
                    
                    for k in loopGroups[i]:
                        lowerDependent += dependence[loopGroups[i][j], k]
                        upperDependent += dependence[k, loopGroups[i][j]]
                        
                    upperMinusLower[j] = upperDependent - lowerDependent
                
                maxDifference = loopGroups[i][upperMinusLower.index(max(upperMinusLower))]
                
                for j in range(0, len(loopGroups[i])):
                    if dependence[maxDifference, loopGroups[i][j]] == 1:
                        toBeAltered.append([maxDifference, loopGroups[i][j]])
    
    return(toBeAltered)

# function to test whether duplicates are present
def duplicatetester(individual):
    allLocations = []
    duplicateLocations = []
    duplicates = False
    for chromosome in individual.chromosomeSet:
        for i in range(0, len(chromosome.tour)):
            if chromosome.tour[i] in allLocations:
                duplicateLocations.append(chromosome.tour[i])
                duplicates = True
            allLocations.append(chromosome.tour[i])
    if duplicates:
        #print('duplicates found! duplicates:', duplicateLocations)
        return(InitializeH2())
    else:
        return(individual)

# function to test whether locations are missing
def missingtester(individual):
    allLocations = []
    missingLocations = []
    missing = False
    for chromosome in individual.chromosomeSet:
        for i in range(0, len(chromosome.tour)):
            allLocations.append(chromosome.tour[i])
    
    for i in range(1, 2 * n + 1):
        if i not in allLocations:
            missingLocations.append(i)
            missing = True
    
    if missing:
        print('missings found! missings:', missingLocations)
        return(InitializeH2())
    else:
        return(individual)

# An implementation of the SGXX
def Crossover(father, mother):
    parent1 = deepcopy(father)
    parent2 = deepcopy(mother)
            
    cutParent1 = randint(1, len(parent1.sequence) - 1)
    cutParent2 = randint(1, len(parent2.sequence) - 1)
    
    childFormation = parent1.sequence[0:cutParent1] + parent2.sequence[cutParent2:len(parent2.sequence)]
    
    #Create the child as an individual by adding all the chromosomes
    child = Individual()
    
    for i in range(0, len(childFormation)):
        child.addChromosomeChild(childFormation[i])
    
    # Make dependence forest of the child and create sets of missing and duplicates
    child.updateDependenceForestChild()
    
    if len(child.missingPickups) > np.sqrt(n):
        return(InitializeH1())
    elif len(child.missingDeliveries) > np.sqrt(n):
        return(InitializeH2())
    
    
    #Check if there is a loop in the dependence forest
    loops = loopChecker(child, [])
    
    impossibleLocations = []
    
    unremovableLoop = False
    unremovablePickup = False
    unremovableDelivery = False
    
    while loops:
        update = False
        i = loops[0][0]
        j = loops[0][1]
        
        
        duplicatesToBeRemoved = []
        chrom = True
        
        #fix with duplicates that i do not have to wait on j anymore
        DupDeliinWait = [val for val in child.chromosomeSet[i].tour if val in child.duplicatesDeliveries]
        AlsoPickInNotWait = [val for val in child.chromosomeSet[j].tour if (val + n) in DupDeliinWait]
        toBeremovedDeli = [x + n for x in AlsoPickInNotWait]
        
        if toBeremovedDeli != []:
            k = toBeremovedDeli[0]
            goalbefore = child.goal
            if removeDuplicates(child, k, chrom, i):
                child.updateDependenceForestChild()
                loops = loopChecker(child, [])
                update = True
                unremovableDelivery = False
            else:
                unremovableDelivery = True
        
        if update:
            continue
        
        DupPickinNotWait = [val for val in child.chromosomeSet[j].tour if val in child.duplicatesPickups]
        AlsoDeliInNotWait = [val for val in child.chromosomeSet[i].tour if (val - n) in DupPickinNotWait]
        toBeremovedPick = [x - n for x in AlsoDeliInNotWait]
        
        if toBeremovedPick != []:
            k = toBeremovedPick[0]
            goalbefore = child.goal
            if removeDuplicates(child, k, chrom, j):
                child.updateDependenceForestChild()
                loops = loopChecker(child, [])
                update = True
                unremovablePickup = False
            else:
                unremovablePickup = True
        
        if update:
            continue
        
        #Fix the connection from j to i, hence remove that j have to wait on i with duplicates
        #removal of duplicate deliveries in j
        DupDeliinWait = [val for val in child.chromosomeSet[j].tour if val in child.duplicatesDeliveries]
        AlsoPickInNotWait = [val for val in child.chromosomeSet[i].tour if (val + n) in DupDeliinWait]
        toBeremovedDeli = [x + n for x in AlsoPickInNotWait]
        
        if toBeremovedDeli != []:
            k = toBeremovedDeli[0]
            goalbefore = child.goal
            if removeDuplicates(child, k, chrom, j):
                child.updateDependenceForestChild()
                loops = loopChecker(child, [])
                update = True
                unremovableDelivery = False
            elif unremovableDelivery:
                return(InitializeH1())
            else:
                unremovableDelivery = True
        
        if update:
            continue
        
        #removal of duplicate pickups in i
        DupPickinNotWait = [val for val in child.chromosomeSet[i].tour if val in child.duplicatesPickups]
        AlsoDeliInNotWait = [val for val in child.chromosomeSet[j].tour if (val - n) in DupPickinNotWait]
        toBeremovedPick = [x - n for x in AlsoDeliInNotWait]
        
        if toBeremovedPick != []:
            k = toBeremovedPick[0]
            goalbefore = child.goal
            if removeDuplicates(child, k, chrom, i):
                child.updateDependenceForestChild()
                loops = loopChecker(child, [])
                update = True
                unremovablePickup = False
            elif unremovablePickup:
                return(InitializeH2())
            else:
                unremovablePickup = True
        
        if update:
            continue
        
        for k in range(0, len(child.chromosomeSet[j].tour)):
            location = child.chromosomeSet[j].tour[k]
            if location <= n and (location + n) in child.chromosomeSet[i].tour:
                child.removeNonDuplicateLocation(child.chromosomeSet[j], location)
                child.updateDependenceForestChild()
                loops = loopChecker(child, [])
                update = True
                unremovableLoop = False
                break
            elif k == len(child.chromosomeSet[j].tour) - 1 and not unremovableLoop:
                loops.pop(0)
                unremovableLoop = True
            elif unremovableLoop:
                return(InitializeH1())
    
    child.updateDependenceForestChild()
    
    chrom = False
    
    if not removeDuplicates(child, child.duplicatesPickups, chrom, -1):
        return(InitializeH1())
    elif not removeDuplicates(child, child.duplicatesDeliveries, chrom, -1):
        return(InitializeH2())
    
    childtours = [[] for x in range(0, len(child.chromosomeSet))]
    
    for i in range(0, len(child.chromosomeSet)):
        childtours[i] = child.chromosomeSet[i].tour

    child.updateDependenceForestChild()
    
    #REINSERTING MISSING LOCATIONS
    while child.missingPickups:
        k = child.missingPickups[0]
        if (k + n) in child.missingDeliveries:
            DeliveryMisses = True
        else:
            DeliveryMisses = False
            indexChromChild = child.chromosomeSet.index(child.visitsChild[k + n][0])
        
        potentialdependence = deepcopy(child.dependence)
        distanceIncrease = [0 for x in range(0, len(child.chromosomeSet))]
        
        for i in range(0, len(child.chromosomeSet)):
            copyOfChrom = deepcopy(child.chromosomeSet[i])
            copyPotentialDependence = deepcopy(potentialdependence)
            
            if not DeliveryMisses:
                copyPotentialDependence[indexChromChild, i] = 1
            else:
                copyPotentialDependence = copyPotentialDependence
            
            if loopChecker(child, copyPotentialDependence) == []:
                DistanceOriginal = copyOfChrom.Delta
                
                if copyOfChrom.addMissingLocation(k):
                    DistanceNew = copyOfChrom.Delta
                    distanceIncrease[i] = DistanceNew - DistanceOriginal
                else:
                    distanceIncrease[i] = MAXVALUE
            else:
                distanceIncrease[i] = MAXVALUE
        
        if all(x == MAXVALUE for x in distanceIncrease):
            newChromosome = Chromosome()
            newChromosome.addLocation(k)
            child.addChromosomeChild(newChromosome)
        else:
            selectedInsertion = distanceIncrease.index(min(distanceIncrease))
            child.addMissingLocation(selectedInsertion, k)
        
        child.updateDependenceForestChild()
    
    #Reinsert missing delivery locations
    while child.missingDeliveries:
        
        k = child.missingDeliveries[0]
        if (k - n) in child.missingPickups:
            PickupMisses = True
            indexChromChild = MAXVALUE
        else:
            PickupMisses = False
            indexChromChild = child.chromosomeSet.index(child.visitsChild[k - n][0])
        
        potentialdependence = deepcopy(child.dependence)
        distanceIncrease = [0 for x in range(0, len(child.chromosomeSet))]
        
        for i in range(0, len(child.chromosomeSet)):
            copyOfChrom = deepcopy(child.chromosomeSet[i])
            copyPotentialDependence = deepcopy(potentialdependence)
            
            if not PickupMisses:
                copyPotentialDependence[i, indexChromChild] = 1
            else:
                copyPotentialDependence = copyPotentialDependence
            
            if loopChecker(child, copyPotentialDependence) == []:
                DistanceOriginal = copyOfChrom.Delta
                if copyOfChrom.addMissingLocation(k):
                    DistanceNew = copyOfChrom.Delta
                    distanceIncrease[i] = DistanceNew - DistanceOriginal
                else:
                    distanceIncrease[i] = MAXVALUE
            else:
                distanceIncrease[i] = MAXVALUE
        
        
        if all(x == MAXVALUE for x in distanceIncrease):
            newChromosome = Chromosome()
            newChromosome.addLocation(k)
            child.addChromosomeChild(newChromosome)
        else:
            selectedInsertion = distanceIncrease.index(min(distanceIncrease))
            child.addMissingLocation(selectedInsertion, k)
        
        child.updateDependenceForestChild()
    
    child = duplicatetester(child)
    child = missingtester(child)
        
    if child.updateIndividual():
        return(child)
    else:
        return(InitializeH1())

# Remove duplicate locations in an individual, if possible.
def removeDuplicates(child, duplicates, chromosomeRemoval, index):
    if chromosomeRemoval:
        chromosome = child.chromosomeSet[index]
        
        if chromosome.removeLocationNewTour(duplicates):
            return(True)
        else:
            return(False)
    
    while duplicates:        
        k = duplicates[0]
        
        Copy0 = deepcopy(child.visitsChild[k][0])
        Copy1 = deepcopy(child.visitsChild[k][1])
        distanceOriginal0 = Copy0.Delta
        distanceOriginal1 = Copy1.Delta
        
        distanceRemoved0 = 0
        distanceRemoved1 = 0
        
        succes0 = True
        succes1 = True
        
        if len(Copy0.tour) != 1:
            succes0 = Copy0.removeLocationNewTour(k)
            distanceRemoved0 = Copy0.Delta
        
        if len(Copy1.tour) != 1:
            succes1 = Copy1.removeLocationNewTour(k)
            distanceRemoved1 = Copy1.Delta
        
        Difference0 = distanceOriginal0 - distanceRemoved0
        Difference1 = distanceOriginal1 - distanceRemoved1
        
        if Difference1 < Difference0 and succes0:
            chromosomeIndex = child.chromosomeSet.index(child.visitsChild[k][0])
            
            if len(child.visitsChild[k][0].tour) == 1:
                child.visitsChild[k][0] = child.visitsChild[k][1]
                child.visitsChild[k][1] = Chromosome()
            
            child.removeDuplicateLocation(chromosomeIndex, k)
            duplicates.pop(0)
        elif succes1:
            chromosomeIndex = child.chromosomeSet.index(child.visitsChild[k][1])
            
            if len(child.visitsChild[k][1].tour) == 1:
                child.visitsChild[k][1] = Chromosome()
            
            child.removeDuplicateLocation(chromosomeIndex, k)
            duplicates.pop(0)
        else:
            duplicates.pop(0)
            return(False)
        
        child.updateDependenceForestChild()
    
    return(True)


# Perform local search on the individual, try to improve it.
def education(individual):
    copyOfIndividual = deepcopy(individual)
        
    for i in range(0, n):
        u = random.randint(1, 2 * n)
        v = random.randint(1, 2 * n)
        
        while u == v:
            v = random.randint(1, 2 * n)
                
        switch = [1, 2, 3, 4, 5]
        
        while switch != []:
            choice = random.choice(switch)
            switch.remove(choice)
            
            if choice == 1:
                result = placeBefore(copyOfIndividual, u, v)
                
                if result[0]:
                    copyOfIndividual = result[1]
                    break
            elif choice == 2:
                result = placeBefore(copyOfIndividual, v, u)
                
                if result[0]:
                    copyOfIndividual = result[1]
                    break
            elif choice == 3:
                result = placeAfter(copyOfIndividual, u, v)
                
                if result[0]:
                    copyOfIndividual = result[1]
                    break
            elif choice == 4:
                result = placeAfter(copyOfIndividual, v, u)
                
                if result[0]:
                    copyOfIndividual = result[1]
                    break
            else:
                result = swap(copyOfIndividual, u, v)
                
                if result[0]:
                    copyOfIndividual = result[1]
                    break
    
    individual = deepcopy(copyOfIndividual)
    
    return(individual)


# Place location u before location v.
def placeBefore(individual, u, v):
    auxiliary = deepcopy(individual)
    
    change = False

    uIndex = auxiliary.visits[u].tour.index(u)
    vIndex = auxiliary.visits[v].tour.index(v)
    
    if auxiliary.visits[u] != auxiliary.visits[v]:
        chromosomeA = deepcopy(auxiliary.visits[u])
        chromosomeB = deepcopy(auxiliary.visits[v])
        
        if chromosomeA.removeLocationAtPosition(uIndex):
            change = chromosomeB.addLocationAtPosition(u, vIndex)
    else:
        chromosomeA = deepcopy(auxiliary.visits[u])
        
        if chromosomeA.removeLocationAtPosition(uIndex):
            if uIndex < vIndex:
                change = chromosomeA.addLocationAtPosition(u, vIndex - 1)
            elif uIndex > vIndex:            
                change = chromosomeA.addLocationAtPosition(u, vIndex)
            else:
                return(False, individual)
                                
    if change:        
        indexChromosomeA = auxiliary.chromosomeSet.index(auxiliary.visits[u])
        auxiliary.chromosomeSet[indexChromosomeA] = deepcopy(chromosomeA)
        
        if auxiliary.visits[u] != auxiliary.visits[v]:
            indexChromosomeB = auxiliary.chromosomeSet.index(auxiliary.visits[v])
            auxiliary.chromosomeSet[indexChromosomeB] = deepcopy(chromosomeB)
        
        if auxiliary.updateIndividual():
            if auxiliary.infeasibility[0] <= individual.infeasibility[0] and auxiliary.goal < individual.goal:
                individual = deepcopy(auxiliary)
                return(True, individual)
            elif auxiliary.infeasibility[0] < individual.infeasibility[0]:
                individual = deepcopy(auxiliary)
                return(True, individual)
    
    return(False, individual)


# Place location u behind location v.
def placeAfter(individual, u, v):
    auxiliary = deepcopy(individual)
    
    change = False

    uIndex = auxiliary.visits[u].tour.index(u)
    vIndex = auxiliary.visits[v].tour.index(v)
    
    if auxiliary.visits[u] != auxiliary.visits[v]:
        chromosomeA = deepcopy(auxiliary.visits[u])
        chromosomeB = deepcopy(auxiliary.visits[v])
        
        if chromosomeA.removeLocationAtPosition(uIndex):
            change = chromosomeB.addLocationAtPosition(u, vIndex + 1)
    else:
        chromosomeA = deepcopy(auxiliary.visits[u])
        
        if chromosomeA.removeLocationAtPosition(uIndex):
            if uIndex < vIndex:
                change = chromosomeA.addLocationAtPosition(u, vIndex)
            elif uIndex > vIndex:            
                change = chromosomeA.addLocationAtPosition(u, vIndex + 1)
            else:
                return(False, individual)
                                
    if change:        
        indexChromosomeA = auxiliary.chromosomeSet.index(auxiliary.visits[u])
        auxiliary.chromosomeSet[indexChromosomeA] = deepcopy(chromosomeA)
        
        if auxiliary.visits[u] != auxiliary.visits[v]:
            indexChromosomeB = auxiliary.chromosomeSet.index(auxiliary.visits[v])
            auxiliary.chromosomeSet[indexChromosomeB] = deepcopy(chromosomeB)
        
        if auxiliary.updateIndividual():
            if auxiliary.infeasibility[0] <= individual.infeasibility[0] and auxiliary.goal < individual.goal:
                individual = deepcopy(auxiliary)
                return(True, individual)
            elif auxiliary.infeasibility[0] < individual.infeasibility[0]:
                individual = deepcopy(auxiliary)
                return(True, individual)
    
    return(False, individual)

# Swap locations u and v
def swap(individual, u, v):
    auxiliary = deepcopy(individual)
    
    changeA = False
    changeB = False

    uIndex = auxiliary.visits[u].tour.index(u)
    vIndex = auxiliary.visits[v].tour.index(v)
    
    if auxiliary.visits[u] != auxiliary.visits[v]:
        chromosomeA = deepcopy(auxiliary.visits[u])
        chromosomeB = deepcopy(auxiliary.visits[v])
        
        if chromosomeA.removeLocationAtPosition(uIndex):
            if chromosomeB.removeLocationAtPosition(vIndex):
                changeA = chromosomeA.addLocationAtPosition(v, uIndex)
                changeB = chromosomeB.addLocationAtPosition(u, vIndex)
    else:
        chromosomeA = deepcopy(auxiliary.visits[u])
        
        if uIndex < vIndex:
            if chromosomeA.removeLocationAtPosition(vIndex):
                if chromosomeA.removeLocationAtPosition(uIndex):
                    changeA = chromosomeA.addLocationAtPosition(v, uIndex)
                    changeA = chromosomeA.addLocationAtPosition(u, vIndex)
        elif uIndex > vIndex:            
            if chromosomeA.removeLocationAtPosition(uIndex):
                if chromosomeA.removeLocationAtPosition(vIndex):
                    changeA = chromosomeA.addLocationAtPosition(u, vIndex)
                    changeA = chromosomeA.addLocationAtPosition(v, uIndex)
        else:
            return(False, individual)
                                
    if changeA and changeB:
        indexChromosomeA = auxiliary.chromosomeSet.index(auxiliary.visits[u])
        auxiliary.chromosomeSet[indexChromosomeA] = deepcopy(chromosomeA)
        
        if auxiliary.visits[u] != auxiliary.visits[v]:
            indexChromosomeB = auxiliary.chromosomeSet.index(auxiliary.visits[v])
            auxiliary.chromosomeSet[indexChromosomeB] = deepcopy(chromosomeB)
        
        if auxiliary.updateIndividual():
            if auxiliary.infeasibility[0] <= individual.infeasibility[0] and auxiliary.goal < individual.goal:
                individual = deepcopy(auxiliary)
                return(True, individual)
            elif auxiliary.infeasibility[0] < individual.infeasibility[0]:
                individual = deepcopy(auxiliary)
                return(True, individual)
    
    return(False, individual)

# This function repairs an infeasible individual.
def repairIndividual(individual):
    if individual.infeasibility[0] > 0:
        lateDeadlines = individual.lateDeadlines
        lateTuples = [(lateDeadlines[x], x) for x in range(1, n + 1) if lateDeadlines[x] > 0]
        lateTuples.sort(reverse = True)
        lateRequests = [x[1] for x in lateTuples]
                
        improvement = True
        
        while improvement:
            improvement = False
            
            operator = deepcopy(individual)
            
            for i in lateRequests:
                succes = False
                
                repairByEducation = locationsBefore(individual, i)
                
                if repairByEducation[0]:
                    individual = deepcopy(repairByEducation[1])
                    improvement = True
                    lateDeadlines = individual.lateDeadlines
                    lateTuples = [(lateDeadlines[x], x) for x in range(1, n + 1) if lateDeadlines[x] > 0]
                    lateTuples.sort(reverse = True)
                    lateRequests = [x[1] for x in lateTuples]
                else:
                    repairByEducation = locationsBefore(individual, i + n)
                    if repairByEducation[0]:
                        individual = deepcopy(repairByEducation[1])
                        improvement = True
                        lateDeadlines = individual.lateDeadlines
                        lateTuples = [(lateDeadlines[x], x) for x in range(1, n + 1) if lateDeadlines[x] > 0]
                        lateTuples.sort(reverse = True)
                        lateRequests = [x[1] for x in lateTuples]
                
                if improvement:
                    break
                else:
                    if individual.visits[i] != individual.visits[i + n]:
                        pickupChromosome = operator.visits[i]
                        deliveryChromosome = operator.visits[i + n]

                        pickupIndex = operator.chromosomeSet.index(operator.visits[i])
                        deliveryIndex = operator.chromosomeSet.index(operator.visits[i + n])

                        if pickupChromosome.addLocation(i + n) and deliveryChromosome.removeLocation(i + n):
                            if deliveryChromosome.addLocation(i) and pickupChromosome.removeLocation(i):
                                operator.chromosomeSet[pickupIndex] = pickupChromosome
                                operator.chromosomeSet[deliveryIndex] = deliveryChromosome
                                if operator.updateIndividual():
                                    succes = True
                    elif individual.visits[i] == individual.visits[i + n]:
                        connectionCreated = createConnection(individual, i)
                        if connectionCreated[0]:
                            individual = deepcopy(repairByEducation[1])
                            succes = True
                            lateDeadlines = individual.lateDeadlines
                            lateTuples = [(lateDeadlines[x], x) for x in range(1, n + 1) if lateDeadlines[x] > 0]
                            lateTuples.sort(reverse = True)
                            lateRequests = [x[1] for x in lateTuples]
                        else:
                            chromosome = operator.visits[i]
                            
                            if chromosome.removeLocation(i) and chromosome.removeLocation(i + n):
                                newChromosome = Chromosome()
                                if newChromosome.addLocation(i) and newChromosome.addLocation(i + n):
                                    operator.addChromosome(newChromosome)
                                    if operator.updateIndividual():
                                        succes = True
                    
                    if succes:
                        if operator.infeasibility[0] < individual.infeasibility[0]:
                            individual = deepcopy(operator)
                            improvement = True
                            lateDeadlines = individual.lateDeadlines
                            lateTuples = [(lateDeadlines[x], x) for x in range(1, n + 1) if lateDeadlines[x] > 0]
                            lateTuples.sort(reverse = True)
                            lateRequests = [x[1] for x in lateTuples]
                
                if improvement:
                    break
    
    return(individual)

# Create a dependence between two chromosomes.
def createConnection(individual, request):
    origin = individual.visits[request]
    chromosomeSequenceIndex = individual.sequence.index(origin)
    chromosomeIndex = individual.chromosomeSet.index(origin)
    
    for i in range(1, chromosomeSequenceIndex + 1):
        chromosome = individual.sequence[chromosomeSequenceIndex - i]
        chromosomeIndex = individual.chromosomeSet.index(chromosome)
        addToEarlier = placeIn(individual, request, chromosomeIndex)
        
        if addToEarlier[0]:
            individual = deepcopy(addToEarlier[1])
            return(True, individual)
        else:
            addToEarlier = placeIn(individual, request + n, chromosomeIndex)
            if addToEarlier[0]:
                individual = deepcopy(addToEarlier[1])
                return(True, individual)
    
    return(False, individual)


# Place location u in chromosome.
def placeIn(individual, u, chromosomeIndex):
    auxiliary = deepcopy(individual)
    
    change = False

    uChromosome = auxiliary.visits[u]
    chromosomeA = deepcopy(auxiliary.chromosomeSet[chromosomeIndex])
    chromosomeB = deepcopy(uChromosome)
    uIndex = auxiliary.chromosomeSet.index(uChromosome)
    
    if chromosomeB.removeLocationNewTour(u):
        change = chromosomeA.addLocation(u)
        
        if change:
            auxiliary.chromosomeSet[chromosomeIndex] = deepcopy(chromosomeA)
            auxiliary.chromosomeSet[uIndex] = deepcopy(chromosomeB)
            
            if auxiliary.updateIndividual():
                if auxiliary.infeasibility[0] <= individual.infeasibility[0] and auxiliary.goal < individual.goal:
                    individual = deepcopy(auxiliary)
                    return(True, individual)
                elif auxiliary.infeasibility[0] < individual.infeasibility[0]:
                    individual = deepcopy(auxiliary)
                    return(True, individual)
    
    return(False, individual)

# Check which locations are visited before a certain location in the chromosome sequence
def locationsBefore(individual, location):
    chromosome = individual.visits[location]
    chromosomeIndex = individual.chromosomeSet.index(chromosome)
    
    upperDependent = [x for x in individual.dependence[chromosomeIndex, ] if x == 1]
    if upperDependent:
        for i in range(0, len(upperDependent)):
            upperDependentLevelTwo = [x for x in individual.dependence[i, ] if x == 1]
        
        upperDependent = list(set(upperDependent + upperDependentLevelTwo))
    
    possibleLocations = []
    
    for i in range(0, len(upperDependent)):
        possibleLocations = possibleLocations + individual.chromosomeSet[i].tour
    
    copyOfIndividual = deepcopy(individual)
    
    for i in possibleLocations:
        switch = [1,2,3]
        while switch:
            choice = random.choice(switch)
            switch.remove(choice)
            
            if choice == 1:
                result = placeBefore(copyOfIndividual, location, i)
                
                if result[0]:
                    individual = deepcopy(result[1])
                    return(True, individual)
            elif choice == 2:
                result = placeAfter(copyOfIndividual, location, i)
                
                if result[0]:
                    individual = deepcopy(result[1])
                    return(True, individual)
            else:
                result = swap(copyOfIndividual, location, i)
                
                if result[0]:
                    individual = deepcopy(result[1])
                    return(True, individual)
    
    return(False, individual)

# The population class contains the intialisation of the intial populations
# and the genetic algorithm.
class Population:
    def __init__(self):
        self.popInfeasible = []
        self.popFeasible = []
        self.parents = []
        self.populationInfeasibility = []
        self.populationSize = 0
        
    def startGeneticAlg(self, populationSize, numberOfBirths):
        self.populationSize = 0.5 * populationSize
        self.initializePop(populationSize)
        #print('Feasible population size: ', len(self.popFeasible))
        
        for i in range(0, numberOfBirths):
            #print(i)
            self.tournament()
            self.mating()
            if len(self.popFeasible) > 0:
                self.popFeasible[0].solutionPrinter()
        
    
    def initializePop(self,numberOfIndividuals):
        for i in range(0,numberOfIndividuals):
            #if i % 10 == 0:
                #print(i)
            
            forceTransfer = np.random.choice([True, False], p = [0.5, 0.5])
            
            if forceTransfer:
                newIndividual = InitializeH2()
            else:
                newIndividual = InitializeH1()
            
            if newIndividual.infeasibility[0] > 0:
                newIndividual = repairIndividual(newIndividual)
            
            if(newIndividual.infeasibility[0] > 0):
                if len(self.popInfeasible) < self.populationSize:
                    heapq.heappush(self.popInfeasible, newIndividual)
                    heapq.heappush(self.populationInfeasibility, newIndividual.infeasibility[0])
                elif newIndividual.infeasibility[0] < max(self.populationInfeasibility):
                    self.popInfeasible.remove(max(self.popInfeasible))
                    heapq.heappush(self.popInfeasible, newIndividual)
                    self.populationInfeasibility.remove(max(self.populationInfeasibility))
                    heapq.heappush(self.populationInfeasibility, newIndividual.infeasibility[0])
            else:
                #print('wow feasible')
                if len(self.popFeasible) < self.populationSize:
                    heapq.heappush(self.popFeasible, newIndividual)
                elif newIndividual < max(self.popFeasible):
                        heapq.heappush(self.popFeasible, newIndividual)
        
        return(self.popFeasible)
    
    #BinaryTournament
    def tournament(self):
        parents = [Individual(), Individual()]
        for fatherMother in range(0, len(parents)):
            feasibleParent = np.random.choice([True, False], p = [0.8, 0.2])
            
            if (feasibleParent or not self.popInfeasible) and self.popFeasible:
                firstDate = random.choice(self.popFeasible)
            elif self.popInfeasible:
                firstDate = random.choice(self.popInfeasible)
            
            feasibleParent = np.random.choice([True, False], p = [0.8, 0.2])
            
            if (feasibleParent or not self.popInfeasible) and self.popFeasible:
                secondDate = random.choice(self.popFeasible)
            elif self.popInfeasible:
                secondDate = random.choice(self.popInfeasible)
            
            if firstDate.infeasibility[0] < secondDate.infeasibility[0]:
                parents[fatherMother] = firstDate
            elif firstDate < secondDate:
                parents[fatherMother] = firstDate
            else:
                parents[fatherMother] = secondDate
            self.parents = parents

    def mating(self):
        self.tournament()
        child = Crossover(self.parents[0], self.parents[1])
        
        if child.infeasibility[0] > 0:
            child = repairIndividual(child)
        
        child = education(child)
        self.survival(child)
    
    
    def survival(self, child):
        if child.infeasibility[0] > 0:
            if child.infeasibility[0] < max(self.populationInfeasibility):
                if child < max(self.popInfeasible):
                    self.popInfeasible.remove(max(self.popInfeasible))
                    heapq.heappush(self.popInfeasible, child)
                    self.populationInfeasibility.remove(max(self.populationInfeasibility))
                    heapq.heappush(self.populationInfeasibility, child.infeasibility[0])
        else:
            #print('Feasible child created')
            if len(self.popFeasible) < self.populationSize:
                heapq.heappush(self.popFeasible, child)
            elif child < max(self.popFeasible):
                self.popFeasible.remove(max(self.popFeasible))
                heapq.heappush(self.popFeasible, child)
    
    
    def populationPrinter(self):
        printPopGoal = [0 for x in range(0, len(self.popFeasible,))]
        for i in range(0, len(self.popFeasible)):
            printPopGoal[i] = self.popFeasible[i].goal
        print(printPopGoal)
    
    
    def infeasiblePopulationPrinter(self):
        printInfeasiblePopGoal = [0 for x in range(0, len(self.popInfeasible,))]
        for i in range(0, len(self.popInfeasible)):
            printInfeasiblePopGoal[i] = (self.popInfeasible[i].goal, self.popInfeasible[i].infeasibility[0])
        print(printInfeasiblePopGoal)


random.seed(1337)
np.random.seed(1337)
pop = Population()
pop.startGeneticAlg(250, 250)
if len(pop.popFeasible) > 0:
    pop.popFeasible[0].solutionPrinter()

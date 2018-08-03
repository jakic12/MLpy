import numpy as np
import random
import matplotlib.pyplot as plt
import datetime
import time

def saveNetwork(network, filename):
	np.save(filename, network)
	out = "network saved as: ", filename, ".npy"
	return out

def openNetwork(filename):
	if filename.endswith(".npy"):
		filename += ".npy"

	network = np.load(filename)
	return network

def ActivationFunction(x, type, derivative = False):
	if type == "sigmoid" and derivative == False:
		return 1 / (1 + np.exp(-x))

	elif type == "sigmoid" and derivative == True:
		return x * (1 - x)

	elif type == "tanh" and derivative == False:
		return np.tanh(x)

	elif type == "tanh" and derivative == True:
		return 1 - pow(np.tanh(x),2)
	else:
		print("unknown activation funtion, please try again...")
		return None


def createNetwork(layerSizeArray, randomRange = [-3,3], stats = True, initializeSynapses = True):
	if len(layerSizeArray) < 3:
		print("Layer count too small (", len(layerSizeArray), "), network must have at least one hidden layer, so layer count must be bigger than 2")
		return None

	if initializeSynapses == True:
		synapse = [np.array([[random.uniform(randomRange[0], randomRange[1]) for _ in range(layerSizeArray[i])] for j in range(layerSizeArray[i+1])]) for i in range(len(layerSizeArray)-1)] #create the synapses array and initialize them
	else:
		synapse = [np.array([[0 for _ in range(layerSizeArray[i])] for j in range(layerSizeArray[i+1])]) for i in range(len(layerSizeArray)-1)] #create the synapses array

	neuron = [np.array([0 for j in range(layerSizeArray[i])]) for i in range(len(layerSizeArray))]

	for i in range(len(layerSizeArray)-1): 
		if len(synapse[i][0]) != len(neuron[i]) or len(synapse[i]) != len(neuron[i+1]): #check for matrix sizes - if they are not correct, dot product wont work
			print(len(synapse[i]),len(neuron[i]),len(synapse[i][0]),len(neuron[i+1]))
			print("Matrix multiplication error, generated matrixes did not work, please try a gain - someone at the development team fucked up...")
			return None

	if stats == True:

		synapseCount = 0
		for i in range(len(layerSizeArray)-1):
			synapseCount += layerSizeArray[i+1]*layerSizeArray[i]#calculato how many synapses a network has

		print("network generated successfully!")
		print("-" * (49 + len(str(len(layerSizeArray))) + len(str(layerSizeArray)) + len(str(synapseCount))))
		print("layer count:", len(layerSizeArray), "layer sizes:", layerSizeArray, "total synapse count:", synapseCount)#stats
		print("-" * (49 + len(str(len(layerSizeArray))) + len(str(layerSizeArray)) + len(str(synapseCount))), end ="\n\n")

	return [synapse, neuron] #return synapse and neurons array

def forwardPropagate(network, dataArray, learning = False, expectedOut = [], activationFunction="sigmoid", ignoreError=False):

	activationFunctions = ["sigmoid"]

	if ignoreError == False and activationFunction not in activationFunctions:
		print("(forwardPropagate): error, activation function '", activationFunction, "' not supported. Supported activation functions: ", activationFunctions, sep="")
		return None

	if ignoreError == False and len(network) != 2:
		print("(forwardPropagate): invalid network, must include synapse and neuron arrays!")
		return None

	synapse = network[0]
	neuron = network[1]

	layerSizeArray = [0 for i in range(len(neuron))]
	for i in range(len(layerSizeArray)):
		layerSizeArray[i] = len(neuron[i]) #get the network size

	if ignoreError == False and learning == True and len(expectedOut) != layerSizeArray[len(layerSizeArray)-1]: #check for expectedOut validity
		print("(forwardPropagate): invalid expected output, expected output size (", len(expectedOut), ") must equal the size of the last layer (", layerSizeArray[len(layerSizeArray)-1],")", sep="")
		return None
	
	if ignoreError == False and len(dataArray) != layerSizeArray[0]:#check validity of the dataArray
		print("(forwardPropagate): invalid data, dataArray length must equal the size of the first layer (", layerSizeArray[0], "!=", len(dataArray),")")
		return None

	neuron[0] = np.array(dataArray) #input data to the first layer of the network

	for i in range(len(layerSizeArray)-1):
		neuron[i+1] = ActivationFunction(np.dot(synapse[i], neuron[i]), activationFunction)

	if learning == True:
		cost = 0
		for i in range(layerSizeArray[len(layerSizeArray)-1]):
			cost += pow(expectedOut[i] - neuron[len(layerSizeArray)-1][i], 2)/2

		return [[synapse, neuron], cost]
	else:
		return neuron[len(neuron)-1]

def backpropagate(network, expectedOut = [], activationFunction = "sigmoid", learningRate = 1, ignoreError = False):

	activationFunctions = ["sigmoid"]

	if ignoreError == False and activationFunction not in activationFunctions:
		print("(backpropagate): error, activation function '", activationFunction, "' not supported. Supported activation functions: ", activationFunctions, sep="")
		return None

	if ignoreError == False and len(network) != 2:
		print("(backpropagate): invalid network, must include synapse and neuron arrays!")
		return None

	synapse = network[0]
	neuron = network[1]

	layerSizeArray = [0 for i in range(len(neuron))]
	for i in range(len(layerSizeArray)):
		layerSizeArray[i] = len(neuron[i]) #get the network size

	if ignoreError == False and len(expectedOut) != layerSizeArray[len(layerSizeArray)-1]: #check for expectedOut validity
		print("(backpropagate): invalid expected output, expected output size (", len(expectedOut), ") must equal the size of the last layer (", layerSizeArray[len(layerSizeArray)-1],")", sep="")
		return None

	deltaJ = [[0 for j in range(layerSizeArray[i])] for i in range(len(layerSizeArray))]

	for i in range(len(layerSizeArray)-1, 0, -1):#loop trough layers
		for j in range(layerSizeArray[i]):#loop trough nerons of that layer

			if i == len(layerSizeArray)-1: #if it is the last layer
				deltaJ[i][j] = (neuron[i][j] - expectedOut[j]) * ActivationFunction(neuron[i][j], activationFunction, True)
			
			if i != len(layerSizeArray)-1:
				sum = 0
				for l in range(layerSizeArray[i+1]):
					sum += deltaJ[i+1][l] * synapse[i].T[j][l]

				deltaJ[i][j] = sum * neuron[i][j] * ActivationFunction(neuron[i][j], activationFunction, True)

			for l in range(layerSizeArray[i-1]):#loop trought neurons of the previous layer
				synapse[i-1][j][l] += deltaJ[i][j] * neuron[i-1][l] * -learningRate

	return [synapse, neuron]

def learn(network, trainingData, expectedOutputData, epocs = -1, costLimit = -1, plot=True, printFrequency=1000):
	if epocs == -1 and costLimit == -1:
		print("(learn): error, required or epoc count or cost limit!")
		return None

	if len(trainingData) != len(expectedOutputData):
		print("(learn): invalid training data, training data length (", len(traningData), ") does not equal expected output length(", len(expectedOutputData), ")", sep="")
		return None

	costArray = []

	start = time.time()

	if epocs > 0:
		for epoc in range(epocs):

			cost = 0
			for j,trainingSample in enumerate(trainingData):
				[network, cost1] = forwardPropagate(network, trainingSample, True, expectedOutputData[j])
				cost += cost1
				network = backpropagate(network, expectedOutputData[j])

				if network == None:
					print("(learn): error occured!")
					break

			if network == None:
					print("(learn): error occured!")
					break

			cost /= len(trainingData)

			if plot == True:
				costArray.append(cost)

			if epoc % printFrequency == 0:
				print("epoc:", epoc, "cost: " ,cost, " - ", str(datetime.timedelta(seconds=time.time()-start)))

	elif costLimit > 0:

		cost = costLimit+1
		epoc = 0
		while cost > costLimit:
			cost = 0
			for j,trainingSample in enumerate(trainingData):
				[network, cost1] = forwardPropagate(network, trainingSample, True, expectedOutputData[j])
				cost += cost1
				network = backpropagate(network, expectedOutputData[j])
				if network == None:
					print("error occured!")
					break

			if network == None:
					print("error occured!")
					break

			cost /= len(trainingData)
				
			if plot == True:
				costArray.append(cost)

			if epoc % printFrequency == 0:
				print("epoc:", epoc, "cost: " ,cost, " - ", str(datetime.timedelta(seconds=time.time()-start)))

			epoc += 1
	else:
		print("costLimit or epoc count cant be 0 or smaller than 0!")
		return None

	end = time.time()

	print("network succesfully learned after :", epocs, "epocs -", str(datetime.timedelta(seconds=end-start)))

	if plot == True:
		plt.plot(costArray)
		plt.xlabel("epoc")
		plt.ylabel("cost")
		plt.show()

	return network
"""
inputs = [[0,0,0], #EXAMPLE!!
		  [0,0,1],
		  [0,1,0],
		  [0,1,1],
		  [1,0,0],
		  [1,0,1],
		  [1,1,0],
		  [1,1,1]]

expectedOutput = [[0,0],[0,1],[1,1],[1,0],[1,0],[1,1],[0,1],[0,0]]

network = createNetwork([3,10,2])
network = learn(network, inputs, expectedOutput, epocs=10000)

while True:
	print(np.round(forwardPropagate(network, [int(x) for x in input().split(",")] ), 0))
"""
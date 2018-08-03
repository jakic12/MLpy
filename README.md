# ML.py
python library to create, use and learn neural networks


First you have to create a network array with the desired size.
example: (network with 3 layers - sizes 3, 10, 2)
```python
network = ML.createNetwork([3,10,2])
```
 
Then you can learn the network using training data.

```python
inputs = [[0,0,0],
	  [0,0,1],
	  [0,1,0],
	  [0,1,1],
	  [1,0,0],
	  [1,0,1],
	  [1,1,0],
	  [1,1,1]]

expectedOutput = [[0,0],[0,1],[1,1],[1,0],[1,0],[1,1],[0,1],[0,0]]

network = learn(network, inputs, expectedOutput, epocs=10000)
```
the learn function will repeat the process of forward and backward propagation, utill it meets a desired goal. The goal can either be:
1. number of iterations (epocs)
```python
network = learn(network, inputs, expectedOutput, epocs=10000)
```
2. desired cost (warning! may create infinite loops if cost limit is too high)
```python
network = learn(network, inputs, expectedOutput, costLimit=0.01)
```


When the network is learned, you can use the network with the forwardPropagate function
```python
print(forwardPropagate(network, [0,1,1]))
```
The forwardPropagate function will return an array, the size of the last layer


You can also manually learn the network:

1. you have to forward propagate a training sample:
```python
[network, cost] = forwardPropagate(network, trainingSample, True, expectedOutputData[0])
```
The forwardPropagate function must be set to learning=True, so it will return the propagated network as well as the cost.

2. backpropagate the network (network must be forwardpropagated on the correct training data before this):
```python
network = backpropagate(network, expectedOutputData[0])
```

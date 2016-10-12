# Neural-network
Machine Learning Exercise 4: multi-class classification > neural network to recognize hand-written digits in Octave/Matlab 

The neural network has 3 layers: an input layer, a hidden layer and an output layer. 
Inputs are pixel values of digit images. Since the images are of size 20x20, this gives us 400 input layer units (excluding the extra bias unit which always outputs +1).
Hidden layer has 25 hidden units. 
Output layer has 10 labels, from 1 to 10 (note that "0" was mapped to label 10).

================================================
Files included:

ex4.m - Octave/MATLAB script

ex4data1.mat - Training set of hand-written digits

displayData.m - Function to help visualize the dataset

fmincg.m - Function minimization routine (similar to fminunc)

sigmoid.m - Sigmoid function

computeNumericalGradient.m - Numerically compute gradients

checkNNGradients.m - Function to help check analytical gradients (i.e. computed thanks to backprop)

debugInitializeWeights.m - Function for initializing weights

predict.m - Neural network prediction function

sigmoidGradient.m - Compute the gradient of the sigmoid function

randInitializeWeights.m - Randomly initialize weights

nnCostFunction.m - Neural network cost function

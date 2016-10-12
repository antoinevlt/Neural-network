# Neural-network
Machine Learning Exercise 4: multi-class classification > neural network to recognize hand-written digits in Octave/Matlab 

The neural network has 3 layers: an input layer, a
hidden layer and an output layer. Inputs are pixel values of
digit images. Since the images are of size 20x20, this gives us 400 input layer
units (excluding the extra bias unit which always outputs +1).

================================================
Files included:

ex4.m - Octave/MATLAB script

ex4data1.mat - Training set of hand-written digits

ex4weights.mat - Neural network parameters for exercise 4

displayData.m - Function to help visualize the dataset

fmincg.m - Function minimization routine (similar to fminunc)

sigmoid.m - Sigmoid function

computeNumericalGradient.m - Numerically compute gradients

checkNNGradients.m - Function to help check analytical gradients (backpropagation)

debugInitializeWeights.m - Function for initializing weights

predict.m - Neural network prediction function

sigmoidGradient.m - Compute the gradient of the sigmoid function

randInitializeWeights.m - Randomly initialize weights

nnCostFunction.m - Neural network cost function

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
          
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% -------------------------------------------------------------
% Feedforward the neural network and return the cost in the
% variable J.
% -------------------------------------------------------------
% add a column of 1 in front of matrix a1 for bias neuron
a1=[ones(m,1) X];

z2=Theta1*a1';
a2=sigmoid(z2);

% add a line of 1 on top of matrix a2 for bias neuron
a2=[ones(1,m); a2];

z3=Theta2*a2;
a3=sigmoid(z3);

% compute matrix Y for which each line corresponds to a training example
% each line contains one 1 at position equals to label y value and other 
% elements are nil 
A=[1:num_labels];
Y=repmat(A,m,1);
Y=(Y==y);

% compute cost function
for i=1:m,
J=J+Y(i,:)*log(a3(:,i))+(1-Y(i,:))*log(1-a3(:,i));
end;
J=-1/m*J;

% compute regularization term of cost function
Theta1sq=Theta1.^2;
Theta1sq=Theta1sq(:,2:end);
Theta2sq=Theta2.^2;
Theta2sq=Theta2sq(:,2:end);
J=J+lambda/(2*m)*(sum(Theta1sq(:))+sum(Theta2sq(:)));


% -------------------------------------------------------------
% Implement the backpropagation algorithm to compute the 
% gradients Theta1_grad and Theta2_grad.
% -------------------------------------------------------------

% compute error for last layer l=3
delta3=a3-Y';

% compute error for hidden layer l=2
delta2=Theta2(:,2:end)'*delta3.*sigmoidGradient(z2);

% compute gradient
Theta1(:,1)=0;
Theta2(:,1)=0;
Theta1_grad=1/m*delta2*a1+lambda/m*Theta1;
Theta2_grad=1/m*delta3*a2'+lambda/m*Theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

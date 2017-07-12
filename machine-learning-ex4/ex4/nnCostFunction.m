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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

yhot = y == 1:max(y);
theta1_weights = Theta1(:, 2:end);
theta2_weights = Theta2(:, 2:end);

% Hypothesis Hx
a1 = [ones(m, 1), X];
z2 =  a1 * Theta1';
a2 = sigmoid(z2);

a2_b = [ones(m, 1), a2];
z3 = a2_b * Theta2';
a3 = sigmoid(z3);

hx = a3;

% Logistic Cost function
pos = yhot .* log(hx);
neg = (1 - yhot) .* log(1 - hx);
cost = -pos - neg;
logit_cost = sum(cost(:)) / m;

% regularization
theta1_cost = sum(sum(theta1_weights .^ 2));
theta2_cost = sum(sum(theta2_weights .^ 2));
reg_cost = lambda / (2 * m) * (theta1_cost + theta2_cost);

% Total Cost
J = logit_cost + reg_cost;





% Compute Gradients
d3 = a3 - yhot;
d2 = d3 * theta2_weights .* sigmoidGradient(z2);
% 10X25           5000X10     5000X25

t1_g = (Theta1_grad + d2' * a1) / m;
%      25X401     5000X25     5000X401
t1_reg = lambda / m * theta1_weights;
t1_reg_b = [zeros(size(t1_reg, 1), 1), t1_reg];
Theta1_grad = t1_g + t1_reg_b;

t2_g = (Theta2_grad + d3' * a2_b) / m;
t2_reg = lambda / m * theta2_weights;
t2_reg_b = [zeros(size(t2_reg, 1), 1), t2_reg];
Theta2_grad = t2_g + t2_reg_b;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

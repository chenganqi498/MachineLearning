function [J grad] = nnCostFunction_test(nn_params, ...
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
L1 = input_layer_size;
L2 = hidden_layer_size;
L3 = num_labels;

new_y = zeros(m, num_labels);

for i = 1 : L3
new_y(:, i) = (y == i);
end

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
new_X = [ones(m, 1) X];
z2 = Theta1 * new_X';
a2 = [ones(1,m);sigmoid(z2)];
z3 = Theta2 * a2;
h = sigmoid(z3);
normal = -1/m * trace(new_y*log(h) + (1-new_y)*log(1-h));
%normal = -1/m * sum(sum(new_y.*log(h)' + (1-new_y).*log(1-h)'));
theta1 = Theta1(:, 2:end);
theta2 = Theta2(:, 2:end);
reg = lambda/(2*m) * (trace(theta1*theta1') + trace(theta2*theta2'));
%reg = lambda/(2*m) * (sum(sum(theta1.*theta1)) + sum(sum(theta2.*theta2)));
J = normal + reg;

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
%er_size
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

d3 = h - new_y';
d2 = Theta2'*d3;
d2 = d2.*sigmoidGradient([ones(1,m); z2]);

temp1 = permute(repmat(d3,1,1,L2+1), [1 3 2]);
temp2 = permute(repmat(a2,1,1,L3), [3 1 2]);
delta_2 = sum(temp1.*temp2, 3);

temp3 = permute(repmat(d2(2:end,:),1,1,L1+1), [1 3 2]);
temp4 = permute(repmat(new_X,1,1,L2), [3 2 1]);
delta_1 = sum(temp3 .* temp4, 3);

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad = 1/m * delta_1 + ...  
       lambda/m * [zeros(size(Theta1,1),1), Theta1(:,2:end)];
Theta2_grad = 1/m * delta_2 + ...
       lambda/m * [zeros(size(Theta2,1),1), Theta2(:,2:end)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

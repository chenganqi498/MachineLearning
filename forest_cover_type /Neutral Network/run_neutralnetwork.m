clear ; close all; clc

%% Load and Normalize Data 
data = load('../../forest_train.csv');
data_val = load('../../forest_validation.csv');
X = data(1:1000, 1:50); y = data(1:1000, 51);
X_val = data_val(1:1000,1:50); y_val = data_val(1:1000, 51);

X_norm = Normalization(X);
X_val_norm = Normalization(X_val);

%% Setup the parameters 
input_layer_size  = 50;  % 50 input parameters
hidden_layer_size = 20;   % 15 hidden units
num_labels = 7;          % 7 labels, from 1 to 7    

%% Initializing Pameters 
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Training data and perform k-fole cross validation
options = optimset('MaxIter', 50);

% Set regularization parameter lambda
lambda = 1;

k = 3; % number of cross validation set
p_cr = zeros(k, 1); % cross validation accuracy

% Generate indices for K-fold cross validation (K = 3)
indices = crossvalind('Kfold', size(X, 1), k);

for i = 1:k    
   
    X_test = X_norm(find(indices == i), :); 
    X_train = X_norm(find(indices ~= i), :);
    y_test = y(find(indices == i));
    y_train = y(find(indices ~= i));
    
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

p_test = predict(Theta1, Theta2, X_test);
p_cr(i) = mean(double(p_test == y_test)) * 100;
end

pred_test = mean(p_cr);
fprintf('\nCross Validation Accuracy: %f\n', pred_test);
pause;

%% Test on validation set and output confusion matrix

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_norm, y, lambda);
                               
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
p_val = predict(Theta1, Theta2, X_val_norm);

for i = 1 : 7
targets(i, :) = (y_val == i);
inputs(i, :) = (p_val == i);
end

plotconfusion(targets,inputs);
fprintf('\nValidation Set Accuracy: %f\n', mean(double(p_val == y_val)) * 100);

%% Regularized logistic regression

%% Initialization and load data
clear ; close all; clc

data = load('../../forest_train.csv');
data_val = load('../../forest_validation.csv');
X = data(1:1000, 1:50); y = data(1:1000, 51);
X_val = data_val(1:1000,1:50); y_val = data_val(1:1000, 51);

%% Normalization
X_norm = Normalization(X);
X_norm = [ones(size(X,1), 1), X_norm]; % Add a column of ones to x
X_val_norm = Normalization(X_val);
X_val_norm = [ones(size(X_val,1), 1), X_val_norm]; % Add a column of ones to x

%% K-fold cross validation
% Set regularization parameter lambda
lambda = 1;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 50);

k = 3; % number of cross validation set
cross_val = Crossvalidation(X_norm, y, k, lambda, options);

fprintf('lambda: %d, Cross Validation Accuracy : %f\n',lambda, cross_val);
out = [lambda, cross_val];
dlmwrite('findparam.csv', out, 'delimiter',',', '-append');
pause;

%% Test on validation set
theta = Logistic(X_norm, y, lambda, options);
proba =  sigmoid(X_val_norm*theta);
[dummy, p_val] = max(proba');

fprintf('lambda: %d, Validation Accuracy : %f\n',lambda, mean(double(p_val' == y_val) * 100));

%% Print confusion matrix
for i = 1 : 7
targets(i, :) = (y_val == i);
inputs(i, :) = (p_val == i);
end

plotconfusion(targets, inputs);

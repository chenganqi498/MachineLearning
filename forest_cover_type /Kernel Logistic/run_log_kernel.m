%% Regularized Kernel Logistic Regression

%% Initialization
clear ; close all; clc

data = load('../../forest_train.csv');
data_val = load('../../forest_validation.csv');
X = data(1:100, 1:50); y = data(1:100, 51);
X_val = data_val(1:100,1:50); y_val = data_val(1:100, 51);

%% K_fold cross validation
X_norm = Normalization(X);
X_val_norm = Normalization(X_val);

% Set regularization parameter lambda to 1 
lambda = 1;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 100);
k = 3; % number of cross validation set

cross_val = Crossvalidation(X_norm, y, k, lambda, options);
fprintf('lambda: %d, Cross Validation Accuracy : %f\n',lambda, cross_val);
out = [lambda, cross_val];
dlmwrite('findparam.csv', out, 'delimiter',',', '-append');
pause;

%% Test on validation set

K = Kernel(X_norm, X_norm);
alpha = Logistic_Kernel(K, y, lambda, options);

K_val = Kernel(X_val_norm, X_norm);
proba =  sigmoid(K_val*alpha);
[dummy, p_val] = max(proba');

fprintf('lambda: %d, Validation Accuracy : %f\n',lambda, mean(double(p_val' == y_val) * 100));

%% Print confusion matrix
for i = 1 : 7
targets(i, :) = (y_val == i);
inputs(i, :) = (p_val == i);
end
plotconfusion(targets, inputs);
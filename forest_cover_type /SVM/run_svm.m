%% Support vector machine

%% Initialization
clear ; close all; clc

%% Load Data
%  The first fifty columns contains the X values and the fifty first column
%  contains the label (y).

data = load('../forest_train.csv');
data_val = load('../forest_validation.csv');

X = data(1:1000, 1:50); y = data(1:1000, 51);
X_norm = Normalization(X);
X_val = data_val(1:100,1:50); y_val = data_val(1:100, 51);
X_val_norm = Normalization(X_val);

% Initialize fitting parameters
class = 7;
initial_theta = zeros(size(X, 2), class);
dist = zeros(size(X, 1), class);
dist_val = zeros(size(X_val, 1), class);

%scan over a certain range of boxconstraint and rbf_sigma
C = 1;
gamma = 2;

%% Prediction of Test set
for type = 1:class
% Optimize
y_new = y(:,1) == type;
svmstruct(type) = svmtrain(X, y_new, 'Kernel_Function', 'rbf', ...
            'boxconstraint', C, 'rbf_sigma', gamma);
end

for type = 1:class
dist(:, type) = distance(X, svmstruct(type));
dist_val(:, type) = distance(X_val, svmstruct(type));
end

[dummy, y_hat] = min(dist, [], 2);
[dummy, y_val_hat] = min(dist_val, [], 2);

fprintf('\nTraining set accuracy: %f\n', nnz(y_hat == y)/size(y,1)*100);
fprintf('\nValidation set accuracy: %f\n', ...
         nnz(y_val_hat == y_val)/size(y_val,1)*100);
   


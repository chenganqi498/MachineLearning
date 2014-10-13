function cross_val = Crossvalidation(X, y, k, lambda, options)

p_cr = zeros(k, 1);
% Generate indices for K-fold cross validation (K = 3)
indices = crossvalind('Kfold', size(X, 1), k);

for i = 1:k
    
    % Initalized overall accuracy
    X_train = X(find(indices ~= i), :);
    y_train = y(find(indices ~= i));
    
    X_test = X(find(indices == i), :); 
    y_test = y(find(indices == i));
    
    theta = Logistic(X_train, y_train, lambda, options);
    
    proba =  sigmoid(X_test*theta);     
    [dummy, p_test] = max(proba');
    p_cr(i) = mean(double(p_test' == y_test)) * 100;
 
end

cross_val = mean(p_cr);
function alpha = Logistic_Kernel(X, y, lambda, options)
% Initialize fitting parameters
    initial_alpha = zeros(size(X, 1), 7);
    alpha = zeros(size(X,1), 7);
    
    for type = 1:7
        y_new = y(:,1) == type;
        % Optimize
        [alpha(:, type), ~] = ...
	    fminunc(@(t)(costFunctionReg(t, X, y_new, lambda)), ...
        initial_alpha(:,type), options);
    end
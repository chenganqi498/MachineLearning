function theta = Logistic(X, y, lambda, options)
% Initialize fitting parameters
    initial_theta = zeros(size(X, 2), 7);
    theta = zeros(size(X,2), 7);
    
    for type = 1:7
        y_new = y(:,1) == type;
        % Optimize
        [theta(:, type), ~] = ...
	    fminunc(@(t)(costFunctionReg(t, X, y_new, lambda)), ...
        initial_theta(:,type), options);
    end
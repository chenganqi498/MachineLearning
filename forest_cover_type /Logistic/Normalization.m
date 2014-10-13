function norm = Normalization(X)
X_10mean = mean(X(:, 1:10), 1);
X_10std = std(X(:, 1:10), 1); 
X_10norm = (X(:, 1:10)- repmat(X_10mean,size(X, 1),1)) ...
           ./repmat(X_10std,size(X, 1),1);

norm = [X_10norm X(:,11:50)];

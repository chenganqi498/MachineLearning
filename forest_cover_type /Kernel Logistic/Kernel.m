function K = Kernel(x1, x2)
% Construct Gaussian Kernel with std = 1.0
std = 1.0;
K = exp(-1*distsq(x1, x2)/(2*std^2));
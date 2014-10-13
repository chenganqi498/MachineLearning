function dist = distance(X, svm_struct)
% shift and scale the data if necessary:   
        for c = 1:size(X, 2)
            X(:,c) = svm_struct.ScaleData.scaleFactor(c) * ...
                (X(:,c) +  svm_struct.ScaleData.shift(c));
        end
    
sv = svm_struct.SupportVectors;
alphaHat = svm_struct.Alpha;
bias = svm_struct.Bias;
kfun = svm_struct.KernelFunction;
kfunargs = svm_struct.KernelFunctionArgs;

dist = (feval(kfun,sv,X,kfunargs{:})'*alphaHat(:)) + bias;
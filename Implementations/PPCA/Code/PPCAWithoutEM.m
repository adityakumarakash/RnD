function [ W, var, X ] = PPCAWithoutEM( Y, q )
% This function finds the Pincipal compoenents using the concepts of
% probabilistic PCA , but without the EM algorithms
% Y -> data
% q -> dimension of PCA needed
% W -> is the same as X = W' Y-Ymean
% X -> projections

% mean estimate
Ymean = mean(Y, 2);

instanceCount = size(Y, 2);
d = size(Y, 1);

% new matrix with mean zeros found
Ynew = Y - repmat(Ymean, 1, instanceCount);

% covariance matrix
covMat = (Ynew * Ynew') / instanceCount;

% find eigen vectors and eigen values
[eigenVectors, eigenValues] = eig(covMat);
eigenValuesDiag = diag(eigenValues);
[~, index] = sort(eigenValuesDiag);
index = flipud(index);

if q >=d 
    q = d - 1;
end

% sorting the eigen vectors
eigenValuesDiag = eigenValuesDiag(index);
eigenVectors = eigenVectors(:, index);

% finding the maximul likelihood estimates using closed form solution
U = eigenVectors(:, [1:q]);
E = diag(eigenValuesDiag(1:q));
var = mean(eigenValuesDiag(q+1:d));

% principal subspace
W = U * sqrt(E - var * eye(q));
M = W' * W + var * eye(q);

% expected X values
X = M\(W' * Ynew);

end


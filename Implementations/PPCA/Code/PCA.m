function [ W, X ] = PCA( Y, q )
% PCA Returns the principal subspace and projection of the data
% Y -> data
% q -> dimension of PCA needed, 0 means find correct 
% W -> is the same as X = W' Y-Ymean
% X -> projections

% find mean
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

if q > 0
    % extracting the required eigen vectors
    W = eigenVectors(:, index(1:q));
    X = W' * Ynew;
else
    % select the top eigen vector based on some threshold
    % TODO
end

end


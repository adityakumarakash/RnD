function [ W, var, X ] = PPCAWithEM( Y, q )
% This function finds the principal components using the Prob PCA with EM
% algorithm. 

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

if q > d - 1
    q = d - 1;
end

% initialization
W = eye(d, q);
Wprev = zeros(d, q);
var = 1;
varPrev = 0;

epsilon = 0.000001;
iteration = 1;
% EM with E and M steps combined
while sum(sum(abs(W-Wprev))) > epsilon || abs(var - varPrev) > epsilon 
    % calculating SW faster
    iteration = iteration + 1;
    SW = (Ynew * (Ynew' * W)) / instanceCount;
    traceS = sum(sum(Ynew .* Ynew));
    Wprev = W;
    M = W' * W + var * eye(q);
    
    W = SW*inv(var * eye(q) + (M \ W')*SW);
    T = SW * (M \ W');
    var = (traceS - trace(T)) / d;
    varPrev = var;
end

M = W' * W + var * eye(q);
X = M \ (W' * Ynew);


end


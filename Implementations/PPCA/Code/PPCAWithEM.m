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

S = Ynew * Ynew' / instanceCount;

% initialization
W = eye(d, q);
Wprev = zeros(d, q);
var = 1;
varPrev = 0;

epsilon = 0.0001;
iteration = 1;

% EM with E and M steps combined

while sum(sum(abs(W-Wprev))) > epsilon || abs(var - varPrev) > epsilon 
    % calculating SW faster
    iteration = iteration + 1;
    SW = (Ynew * (Ynew' * W)) / (instanceCount*1.0);
    traceS = sum(sum(Ynew .* Ynew)) / (instanceCount*1.0);
    Wprev = W;
    M = W' * W + var * eye(q);
    
    W = SW*inv(var * eye(q) + inv(M)*W'*SW);
    T = SW * inv(M) * W';
    varPrev = var;
    var = (traceS - trace(T)) / d;

%     % Not optimized
%     Wprev = W;
%     W = S*W*inv(var*eye(q)+inv(W'*W+var*eye(q))*W'*S*W);
%     varPrev = var;
%     var = trace(S-S*Wprev*inv(Wprev' * Wprev+var*eye(q))*W')/d;

end

fprintf('no of iterations = %d\n', iteration);

M = W' * W + var * eye(q);
X = M \ (W' * Ynew);


end


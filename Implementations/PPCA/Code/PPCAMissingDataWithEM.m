function [ W, var, X ] = PPCAMissingDataWithEM( Y, q, M )
% This function finds the principal components when some data is missing
%   M is the missing data indicator


d = size(Y, 1);
instanceCount = size(Y, 2);
missingInd = (sum(M) ~= 0);

% mean estimation
YMean = sum(Y .* M, 2) ./ sum(M, 2);
YMeanMat = repmat(YMean, 1, instanceCount);
YNew = Y .* (1 - M) - YMeanMat;

% initialization of parameters
W = eye(d, q);
WPrev = zeros(d, q);
varPrev = 0;
var = 1;
epsilon = 0.00001;
iteration = 1;
XEst = zeros(q, instanceCount); % This would be updated in each iteration

% EM loop
while sum(sum(abs(W - WPrev))) > epsilon || abs(var - varPrev) > epsilon
    iteration = iteration + 1;
    YEst = YNew;
    for i = 1 : instanceCount
        if missingInd(i) == 1
            [x, tEst] = qrSolution(W, Y(:, i), YMean, M(:, i));
            XEst(:, i) = x;
            YEst(:, i) = tEst - YMean;
        end
    end
    
    SW = (YEst * (YEst' * W)) / (instanceCount*1.0);
    traceS = sum(sum(YEst .* YEst)) / (instanceCount*1.0);
    WPrev = W;
    M = W' * W + var * eye(q);
    
    W = SW*inv(var * eye(q) + inv(M)*W'*SW);
    T = SW * inv(M) * W';
    varPrev = var;
    var = (traceS - trace(T)) / d;

end


M = W' * W + var * eye(q);
X = M \ (W' * Ynew);

X(:, missingInd) = XEst(:, missingInd);

end


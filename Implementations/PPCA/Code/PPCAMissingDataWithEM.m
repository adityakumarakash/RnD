function [ W, var, X ] = PPCAMissingDataWithEM( Y, q, Miss )
% This function finds the principal components when some data is missing
%   M is the missing data indicator


d = size(Y, 1);
instanceCount = size(Y, 2);
missingInd = (sum(Miss) ~= 0);

% mean estimation
YMean = sum(Y.*(1-Miss), 2) ./ sum((1-Miss), 2);
YMeanMat = repmat(YMean, 1, instanceCount);
YNew = Y .* (1 - Miss) - YMeanMat;

% initialization of parameters
%[W,~] = PCA(Y(:, sum(Miss) == 0), 2);
W = rand(d, q);
norm = sum(W);
W = W ./ norm(ones(d, 1), :);

WPrev = zeros(d, q);
varPrev = 0.0;
var = 1.0;
epsilon = 0.01;
iteration = 1;
X = zeros(q, instanceCount); % This would be updated in each iteration

% EM loop
while sum(sum(abs(W-WPrev)))/sum(sum(abs(WPrev))) > epsilon || abs(var-varPrev) > epsilon
    iteration = iteration + 1;
    YEst = YNew;
    % E step
    invM = inv(W' * W + var * eye(q));
    tic
    for i = 1 : instanceCount
        if missingInd(i) == 1
            [x, t] = qrSolution(W, Y(:, i), YMean, Miss(:, i));
            X(:, i) = x;
            YEst(:, i) = t - YMean;
        else
            X(:, i) = invM * W' * YNew(:, i);
        end
        if mod(i, 100) == 0
            toc
            tic
        end
    end

    % M Step
    
    WPrev = W;
    W = (YEst * X') * inv(var * instanceCount * invM + X * X');
    varPrev = var;    
    var = (sum(sum(YEst .* YEst)) - 2 * trace(X' * W' * YEst) + trace(instanceCount * var * invM * W' * W + X * X' * W' * W)) / (instanceCount * d);
    YNew = YEst;
    
end
fprintf('No of iterations = %d\n', iteration);

M = W' * W + var * eye(q);
X = M \ (W' * YNew);

end


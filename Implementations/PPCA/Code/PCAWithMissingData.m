function [ W, X ] = PCAWithMissingData( Y, q, Miss )
% This function finds PCA with missing data without EM algorithm
% We use alternate optimization in this case.

d = size(Y, 1);
instanceCount = size(Y, 2);
missingInd = (sum(Miss) ~= 0);

% initialize the missing value, mean, W
YMean = sum(Y.*(1-Miss), 2) ./ sum((1-Miss), 2);
Ytemp = repmat(YMean, 1, instanceCount) .* Miss;
%Ytemp = zeros(size(Y));
%Ytemp = randi(max(max(Y)), size(Y, 1), size(Y, 2));
Y(Miss == 1) = Ytemp(Miss == 1);

YMean = mean(Y, 2);
[W, ~] = PCA(Y, q);

% previous variables
YMeanPrev = zeros(size(YMean));
WPrev = zeros(size(W));
epsilon = 0.0001;
iteration = 0;

% Alternate maximization
while sum(sum(abs(W - WPrev))) > epsilon || sum(abs(YMean - YMeanPrev)) > epsilon
    WPrev = W;
    YMeanPrev = YMean;
    iteration = iteration + 1;
    
    % minimize the error for |Y-XW| for the missing data case
    YEst = Y;
    for i = 1 : instanceCount
        if missingInd(i) == 1
            [~, t] = qrSolution(W, Y(:, i), YMean, Miss(:, i));
            YEst(:, i) = t - YMean;
        end
    end
    Y = YEst;
    
    % estimate the X and W from the estimate values of Y
    [W, ~] = PCA(Y, q);
    YMean = mean(Y, 2);    
end

fprintf('Iteration count = %d\n', iteration);
[W, X] = PCA(Y, q);
end


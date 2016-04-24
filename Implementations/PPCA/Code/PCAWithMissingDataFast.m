function [ W, X ] = PCAWithMissingDataFast( Y, q, Miss )
% This function finds PCA with missing data without EM algorithm
% We use alternate optimization in this case.

d = size(Y, 1);
instanceCount = size(Y, 2);
missingInd = (sum(Miss) ~= 0);

% initialize the missing value, mean, W
YMean = sum(Y.*(1-Miss), 2) ./ sum((1-Miss), 2);
Ytemp = repmat(YMean, 1, instanceCount);
%Ytemp = zeros(size(Y));
%Ytemp = randi(max(max(Y)), size(Y, 1), size(Y, 2));
Y(Miss == 1) = Ytemp(Miss == 1);

YMean = mean(Y, 2);
[W, ~] = PCA(Y, q);

% previous variables
YMeanPrev = zeros(size(YMean));
WPrev = zeros(size(W));
epsilon = 0.01;
iteration = 0;
WRef = -1 * eye(d);

% Alternate maximization
tic
while sum(sum(abs(W - WPrev)))/sum(sum(abs(WPrev))) > epsilon || sum(abs(YMean - YMeanPrev))/sum(sum(abs(YMeanPrev))) > epsilon
    WPrev = W;
    YMeanPrev = YMean;
    iteration = iteration + 1;
    if iteration > 1
        break
    end
    % minimize the error for |Y-XW| for the missing data case
    YEst = Y;
    parfor i = 1 : instanceCount
        if missingInd(i) == 1
%             [~, t] = qrSolution(W, Y(:, i), YMean, Miss(:, i));
%             YEst(:, i) = t;
            WNew = [W WRef(:, Miss(:, i) == 1)];
            xNew = pinv(WNew) * (Y(:, i) .* (1 - Miss(:, i)) - YMean);
            temp = YEst(:, i);
            temp(Miss(:, i) == 1) = xNew(q + 1 : q + sum(Miss(:, i)));
            YEst(:, i) = temp;
        end
    end
    Y = YEst;
    
    % estimate the X and W from the estimate values of Y
    [W, ~] = PCA(Y, q);
    YMean = mean(Y, 2);    
end
toc
fprintf('Iteration count = %d\n', iteration);
[W, X] = PCA(Y, q);
end


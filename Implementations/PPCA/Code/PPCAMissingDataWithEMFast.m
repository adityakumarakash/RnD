function [ W, var, X ] = PPCAMissingDataWithEMFast( Y, q, Miss )
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
iteration = 0;
X = zeros(q, instanceCount); % This would be updated in each iteration
WRef = -1 * eye(d);

tic
% EM loop
while sum(sum(abs(W-WPrev)))/sum(sum(abs(WPrev))) > epsilon || abs(var-varPrev) > epsilon
    iteration = iteration + 1;
    YEst = YNew;
%     if iteration > 1
%         break
%     end
    % E step
    invM = inv(W' * W + var * eye(q));
   
%     parfor i = 1 : instanceCount
%         if missingInd(i) == 1
%             WRef = -1 * eye(d);
%             WNew = [W WRef(:, Miss(:, i) == 1)];
%             xNew = pinv(WNew) * (Y(:, i) .* (1 - Miss(:, i)) - YMean);
%             X(:, i) = xNew(1 : q);
%             temp = YEst(:, i);
%             temp(Miss(:, i) == 1) = xNew(q + 1 : q + sum(Miss(:, i))) - YMean(Miss(:, i) == 1);
%             YEst(:, i) = temp;
%         else
%             X(:, i) = invM * W' * YNew(:, i);
%         end     
%     end
    YEst = Y;
    parfor i = 1 : instanceCount
        if missingInd(i) == 1
            WNew = [W WRef(:, Miss(:, i) == 1)];
            xNew = pinv(WNew) * (Y(:, i) .* (1 - Miss(:, i)) - YMean);
            X(:, i) = xNew(1 : q);
            temp = YEst(:, i);
            temp(Miss(:, i) == 1) = xNew(q + 1 : q + sum(Miss(:, i)));
            YEst(:, i) = temp;
        end     
    end
    YEst = YEst - YMean(:, ones(1, instanceCount));
    X(:, missingInd == 0) = invM * W' * YNew(:, missingInd == 0);
   
    % M Step
    WPrev = W;
    W = (YEst * X') * inv(var * instanceCount * invM + X * X');
    varPrev = var;    
    var = (sum(sum(YEst .* YEst)) - 2 * trace(X' * W' * YEst) + trace(instanceCount * var * invM * W' * W + X * X' * W' * W)) / (instanceCount * d);
    YNew = YEst;
    
end
toc
fprintf('No of iterations = %d\n', iteration);
M = W' * W + var * eye(q);
X = M \ (W' * YNew);

end


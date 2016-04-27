function [W, X] = StandardPCAWithMissingData(Y, q, Miss)
%UNTITLED Summary of this function goes here
%   q is the hidden dimension
% The process is to estimate the mew, Cov and missing data using alternate
% minimization of negative of log likelihood.

d = size(Y, 1);
instanceCount = size(Y, 2);
YPrev = zeros(d, instanceCount);
mewPrev = zeros(d, 1);
CovPrev = zeros(d, d);

% initialization
mew = sum(Y.*(1-Miss), 2) ./ sum(1 - Miss, 2);  
mewRep = repmat(mew, 1, instanceCount);
Y(Miss==1) = mewRep(Miss==1);
Cov = Y * Y' / instanceCount;
epsilon = 0.0001;
iteration = 0;

while sum(abs(mew-mewPrev)) > epsilon || sum(sum(abs(Cov-CovPrev))) > epsilon || sum(sum(abs(Y(Miss==1) - YPrev(Miss==1)))) > epsilon
    % update the loop conditions
    iteration = iteration + 1;
    YPrev = Y;
    CovPrev = Cov;
    mewPrev = mew;
    
    % update the values
    mew = mean(Y, 2);
    mewRep = repmat(mew, 1, instanceCount);
    Y(Miss==1) = mewRep(Miss==1);
    Cov = Y * Y' / instanceCount;
    
    if iteration > 50
        break;
    end
end
fprintf('Iteration count = %d\n', iteration);

% get the principal components for the data
[W, X] = PCA(Y, q);

end


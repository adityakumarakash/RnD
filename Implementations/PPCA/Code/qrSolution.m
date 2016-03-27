function [ x, tEstimated ] = qrSolution( W, t, mew, M)
% Solves for min |Wx-t+mew|
% The solution is based on QR factorization of the particular constraint
% equation, M is the indicator for the missing data in t, 0 not missing , 1
% means missing

[d,q] = size(W);

% the RHS of the linear equation
b = t .* (1-M) - mew;

% forming the new constraint matrix
Wnew = -1*eye(d);
Wnew = [W Wnew(:, M==1)];


% QR factorization to solve Wnew * x = b
[Q, R] = qr(Wnew, 'econ');
y = Q' * b;
xNew = R\y;
 
% xNew = linsolve(Wnew, b);

% separating x and t parts
x = xNew(1:q);

tEstimated = t;
tEstimated(M == 1) = xNew([zeros(1,q) ones(1,sum(M))]' == 1);

end


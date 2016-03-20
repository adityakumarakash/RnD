%% Implementation of PCA . PPCA - EM and general
% load the data
load('../../Data/virus3.dat');
Y = virus3';

%% PCA implementation on entire data
[W, X] = PCA(Y, 2);
figure;
scatter(X(1, :), X(2, :));
T = cellstr(num2str([1:size(X, 2)]'));
text(X(1, :) + 0.1, X(2, :) + 0.1, T);
title('PCA on tobamovirus');

%% PPCA implementation without EM
[W, X] = PPCAWithoutEM(Y, 2);
figure;
scatter(X(1, :), X(2, :));
T = cellstr(num2str([1:size(X, 2)]'));
text(X(1, :) + 0.1, X(2, :) + 0.1, T);
title('PPCA on tobamovirus without using EM algorithm');

%% PPCA with EM 


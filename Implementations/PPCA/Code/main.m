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
[W, var, X] = PPCAWithoutEM(Y, 2);
figure;
scatter(X(1, :), X(2, :));
T = cellstr(num2str([1:size(X, 2)]'));
text(X(1, :) + 0.1, X(2, :) + 0.1, T);
title('PPCA on tobamovirus without using EM algorithm');

%% PPCA with EM 
[W, var, X] = PPCAWithEM(Y, 2); 
figure;
scatter(X(1, :), X(2, :));
T = cellstr(num2str([1:size(X, 2)]'));
text(X(1, :) + 0.1, X(2, :) + 0.1, T);
title('PPCA on tobamovirus with using EM algorithm');


%% PPCA with missing data and EM
M = rand(size(Y)) > 0.9;
fprintf('Missing values count = %d\n', sum(sum(M)));
disp(sum(M));
fprintf('Data points with no missing elements - ');
MInstance = sum(M);

for i = 1 : size(Y, 2)
    if MInstance(i) == 0
        fprintf('%d ', i);
    end
end

[W, var, X] = PPCAMissingDataWithEM(Y, 2, M); 
figure;
scatter(X(1, :), X(2, :));
T = cellstr(num2str([1:size(X, 2)]'));
text(X(1, :) + 0.1, X(2, :) + 0.1, T);
title('PPCA on tobamovirus missing data with using EM algorithm');

%% PCA with missing data and no EM
[W, X] = PCAWithMissingData(Y, 2, M);
figure;
scatter(X(1, :), X(2, :));
T = cellstr(num2str([1:size(X, 2)]'));
text(X(1, :) + 0.1, X(2, :) + 0.1, T);
title('PPCA on tobamovirus missing data without EM');


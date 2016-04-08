%% This code is to analyse PPCA with EM formulations for MNIST data set
% Testing the EM formulation of PPCA on missing data against the modified
% version of PCA which handles missing data
% Classification is done using Mahalanobis distance from each class

%% Load the dataset - train and test
imagesTrain = loadMNISTImages('../../Data/mnist/train-images.idx3-ubyte');
labelsTrain = loadMNISTLabels('../../Data/mnist/train-labels.idx1-ubyte');
labelsTrain = labelsTrain';
imagesTest = loadMNISTImages('../../Data/mnist/t10k-images.idx3-ubyte');
labelsTest = loadMNISTLabels('../../Data/mnist/t10k-labels.idx1-ubyte');


%% Initializations
q = 133;      % latent space dimension, 133 works best for PCA standard
d = size(imagesTrain, 1);       % observed space dimension

%% Mahalanobis distance from original data
% Dist = zeros(size(imagesTest, 2), 10);
% for digit = 0 : 9
%     Y = imagesTrain(:, labelsTrain == digit);
%     Dist(:, digit + 1) = mahal(imagesTest', Y');
% end
% [~, predictedLabels] = min(Dist, [], 2);
% predictedLabels = predictedLabels - 1;
% sum(predictedLabels == labelsTest)


%% Standard PCA
Dist = zeros(size(imagesTest, 2), 10);
for digit = 0 : 9
    Y = imagesTrain(:, labelsTrain == digit);
    [W, X] = PCA(Y, q);
    mew = mean(Y, 2);
    XTest = W' * (imagesTest - mew(:, ones(1, size(imagesTest, 2))));
    Dist(:, digit + 1) = mahal(XTest', X');
end

[~, predictedLabels] = min(Dist, [], 2);
predictedLabels = predictedLabels - 1;
fprintf('Accurracy with Standard PCA, with q = %d, is %f\n', q, (sum(predictedLabels == labelsTest))*100/size(labelsTest, 1));

%% PPCA without EM
Dist = zeros(size(imagesTest, 2), 10);
for digit = 0 : 9
    Y = imagesTrain(:, labelsTrain == digit);
    %[W, var, X] = PPCAWithEM(Y, q); figure; scatter(X(1,:), X(2, :), '.r');
    [W, var, X] = PPCAWithoutEM(Y, q); %figure; scatter(X(1, :), X(2, :), '.r')
    M = W' * W + var * eye(q);
    mew = mean(Y, 2);
    XTest = M\(W' * (imagesTest - mew(:, ones(1, size(imagesTest, 2)))));
    Dist(:, digit + 1) = mahal(XTest', X');
end

[~, predictedLabels] = min(Dist, [], 2);
predictedLabels = predictedLabels - 1;
fprintf('Accurracy with PPCA without EM, with q = %d, is %f\n', q, (sum(predictedLabels == labelsTest))*100/size(labelsTest, 1));

%% PPCA with EM
Dist = zeros(size(imagesTest, 2), 10);
for digit = 0 : 9
    Y = imagesTrain(:, labelsTrain == digit);
    [W, var, X] = PPCAWithEM(Y, q);
    M = W' * W + var * eye(q);
    mew = mean(Y, 2);
    XTest = M\(W' * (imagesTest - mew(:, ones(1, size(imagesTest, 2)))));
    Dist(:, digit + 1) = mahal(XTest', X');
end

[~, predictedLabels] = min(Dist, [], 2);
predictedLabels = predictedLabels - 1;
fprintf('Accurracy with PPCA with EM, with q = %d, is %f\n', q, (sum(predictedLabels == labelsTest))*100/size(labelsTest, 1));


%% Missing Data Case
M = rand(size(Y)) > 0.9;
fprintf('Missing values count = %d\n', sum(sum(M)));

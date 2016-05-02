%% This code is to analyse PPCA with EM formulations data sets
% Testing the EM formulation of PPCA on missing data against the modified
% version of PCA which handles missing data
% Classification is done using Mahalanobis distance from each class

%% Load the dataset - train and test

% loading usps dataset
load('../../Data/usps/usps_all.mat');   % images loaded in data
imageCount = size(data, 2) * size(data, 3);
imageSize = size(data, 1);
imagesData = zeros(imageSize, imageCount);
labelsData = zeros(1, imageCount);
digits = size(data, 3);
for i = 1 : size(data, 3)
    for j = 1 : size(data, 2)
        imagesData(:, (i - 1) * size(data, 2) + j) = data(:, j, i);
        labelsData(:, (i - 1) * size(data, 2) + j) = i;
    end
end
fId = fopen('../Outputs/uspsOutput.txt', 'a');

% % loading the alpha digit datasets
% load('../../Data/alphaDigits/binaryalphadigs.mat');
% % all classes have 39 images
% imageCount = size(dat, 1) * size(dat, 2);
% imageSize = size(dat{1,1}, 1) * size(dat{1,1}, 2);
% imagesData = zeros(imageSize, imageCount);
% labelsData = zeros(1, imageCount);
% digits = size(dat, 1);
% for i = 1 : size(dat, 1)
%     for j = 1 : size(dat, 2)
%         imagesData(:, (i - 1) * size(dat, 2) + j) = reshape(dat{i, j}', imageSize, 1);
%         labelsData(:, (i - 1) * size(dat, 2) + j) = i;
%     end
% end
% fId = fopen('../Outputs/alphaDigitOutput.txt', 'a');


% divide into train and test samples
trainFraction = 0.7;
trainCount = int32(trainFraction * imageCount);
sampleIndex = randsample(imageCount, trainCount);
trainSample = zeros(1, imageCount);
trainSample(sampleIndex) = 1;

imagesTrain = imagesData(:, trainSample==1);
labelsTrain = labelsData(:, trainSample==1)';
imagesTest = imagesData(:, trainSample==0);
labelsTest = labelsData(:, trainSample==0)';



%% Initializations
q = 100;      % latent space dimension, 133 works best for PCA standard for MNIST
d = size(imagesTrain, 1);       % observed space dimension
fprintf(fId, '\n------------------------------------------------------\n');

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
Dist = zeros(size(imagesTest, 2), digits);
for digit = 1 : digits
    Y = imagesTrain(:, labelsTrain == digit);
    [W, X] = PCA(Y, q);
    mew = mean(Y, 2);
    XTest = W' * (imagesTest - mew(:, ones(1, size(imagesTest, 2))));
    Dist(:, digit) = mahal(XTest', X');
end

[~, predictedLabels] = min(Dist, [], 2);
fprintf(fId, 'Accurracy with Standard PCA, with q = %d, is %f\n', q, (sum(predictedLabels == labelsTest))*100/size(labelsTest, 1));

%% PPCA without EM
Dist = zeros(size(imagesTest, 2), digits);
for digit = 1 : digits
    Y = imagesTrain(:, labelsTrain == digit);
    %[W, var, X] = PPCAWithEM(Y, q); figure; scatter(X(1,:), X(2, :), '.r');
    [W, var, X] = PPCAWithoutEM(Y, q); %figure; scatter(X(1, :), X(2, :), '.r')
    M = W' * W + var * eye(q);
    mew = mean(Y, 2);
    XTest = M\(W' * (imagesTest - mew(:, ones(1, size(imagesTest, 2)))));
    Dist(:, digit) = mahal(XTest', X');
end

[~, predictedLabels] = min(Dist, [], 2);
fprintf(fId, 'Accurracy with PPCA without EM, with q = %d, is %f\n', q, (sum(predictedLabels == labelsTest))*100/size(labelsTest, 1));

%% PPCA with EM
Dist = zeros(size(imagesTest, 2), digits);
for digit = 1 : digits
    Y = imagesTrain(:, labelsTrain == digit);
    [W, var, X] = PPCAWithEM(Y, q);
    M = W' * W + var * eye(q);
    mew = mean(Y, 2);
    XTest = M\(W' * (imagesTest - mew(:, ones(1, size(imagesTest, 2)))));
    Dist(:, digit) = mahal(XTest', X');
end

[~, predictedLabels] = min(Dist, [], 2);
fprintf(fId, 'Accurracy with PPCA with EM, with q = %d, is %f\n', q, (sum(predictedLabels == labelsTest))*100/size(labelsTest, 1));
 

%% Missing Data Case using EM
missPercent = 0.01;
fprintf(fId, 'Miss Percentage = %f\n', missPercent*100);
MissIndex = rand(size(imagesTrain)) > 1 - missPercent;
fprintf(fId, 'Missing values count = %d\n', sum(sum(MissIndex)));

Dist = zeros(size(imagesTest, 2), digits);
for digit = 1 : digits
    Y = imagesTrain(:, labelsTrain == digit);
    Miss = MissIndex(:, labelsTrain == digit);
    [W, var, X] = PPCAMissingDataWithEMFast(Y, q, Miss);
    M = W' * W + var * eye(q);
    mew = mean(Y, 2);
    XTest = M\(W' * (imagesTest - mew(:, ones(1, size(imagesTest, 2)))));
    Dist(:, digit) = mahal(XTest', X');
end

[~, predictedLabels] = min(Dist, [], 2);
fprintf(fId, 'Accurracy with PPCA Missing data with EM, with q = %d, is %f\n', q, (sum(predictedLabels == labelsTest))*100/size(labelsTest, 1));
% 
% 
% %% Missing Data without EM case
% % M is the missing data index
% Dist = zeros(size(imagesTest, 2), digits);
% for digit = 1 : digits
%     Y = imagesTrain(:, labelsTrain == digit);
%     Miss = MissIndex(:, labelsTrain == digit);
%     [W, X] = PCAWithMissingDataFast(Y, q, Miss);
%     mew = mean(Y, 2);
%     XTest = W\(imagesTest - mew(:, ones(1, size(imagesTest, 2))));
%     Dist(:, digit) = mahal(XTest', X');
% end
% 
% [~, predictedLabels] = min(Dist, [], 2);
% fprintf(fId, 'Accurracy with PCA Missing data without EM, with q = %d, is %f\n', q, (sum(predictedLabels == labelsTest))*100/size(labelsTest, 1));

% %% Standard PCA with Missing data
% % In this case of missing data, we minimize the fitting of gaussian
% % likelihood to he complete data, we alternatively maximize the mew, Cov
% % and the missing data estimation
% 
% % M is the missing data index
% Dist = zeros(size(imagesTest, 2), digits);
% for digit = 1 : digits
%     Y = imagesTrain(:, labelsTrain == digit);
%     Miss = MissIndex(:, labelsTrain == digit);
%     [W, X] = StandardPCAWithMissingData(Y, q, Miss);
%     mew = mean(Y, 2);
%     XTest = W\(imagesTest - mew(:, ones(1, size(imagesTest, 2))));
%     Dist(:, digit) = mahal(XTest', X');
% end
% 
% [~, predictedLabels] = min(Dist, [], 2);
% fprintf(fId, 'Accurracy with Standard PCA Missing data, with q = %d, is %f\n', q, (sum(predictedLabels == labelsTest))*100/size(labelsTest, 1));

%% This code is to analyse PPCA with EM formulations for MNIST data set
% Testing the EM formulation of PPCA on missing data against the modified
% version of PCA which handles missing data
% Classification is done using Mahalanobis distance from each class

%% Load the dataset - train and test
imagesTrain = loadMNISTImages('../../Data/mnist/train-images.idx3-ubyte');
labelsTrain = loadMNISTLabels('../../Data/mnist/train-labels.idx1-ubyte');
imagesTest = loadMNISTImages('../../Data/mnist/t10k-images.idx3-ubyte');
labelsTest = loadMNISTLabels('../../Data/mnist/t10k-labels.idx1-ubyte');

%% Standard PCA

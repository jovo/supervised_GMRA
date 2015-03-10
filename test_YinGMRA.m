disp('testing the Y in GMRA...')

%% Add Directories to the path
dir = fileparts(mfilename('fullpath'));
cd(dir);
pwd
cd /home/collabor/yb8/supervised_GRMA
pwd
addpath(genpath(pwd));
% cd /home/collabor/yb8/GMRA
% pwd
% addpath(genpath(pwd));
% cd /home/collabor/yb8/DiffusionGeometry
% pwd
% addpath(genpath(pwd));
% cd /home/collabor/yb8/supervised_GRMA/TempTest
% pwd
% addpath(genpath(pwd));
% cd /home/collabor/yb8/supervised_GRMA/data
% pwd
% addpath(genpath(pwd));
cd /home/collabor/yb8/LOL
pwd
addpath(genpath(pwd));
cd(dir);

%% Pick a data set

pDataSetNames  = {'MNIST_HardBinary_T60K_t10K', 'MNIST_HardBinary_T5.0K_t5.0K', 'MNIST_HardBinary_T2.5K_t2.5K', 'MNIST_EasyBinary_T2.5K_t2.5K', 'MNIST_EasyBinary_T0.8K_t0.8K', 'MNIST_EasyBinary_T0.7K_t0.7K', 'MNIST_EasyBinary_T0.6K_t0.6K', 'MNIST_EasyBinary_T0.5K_t0.5K', 'MNIST_EasyBinary_T0.4K_t0.4K', 'MNIST_EasyBinary_T0.3K_t0.3K', 'MNIST_EasyBinary_T0.2K_t0.2K', 'MNIST_EasyTriple_T0.6K_t0.6K', 'MNIST_EasyTriple_T0.3K_t0.3K', 'MNIST_EasyBinary_T10_t10', 'Gaussian_2', 'FisherIris' };

fprintf('\n Data Sets:\n');
for k = 1:length(pDataSetNames),
    fprintf('\n [%d] %s',k,pDataSetNames{k});
end;
fprintf('\n\n  ');

pDataSetIdx = input('Pick a data set to test: \n');

%% Load the data set

[X, TrainGroup, Labels] = LoadData(pDataSetNames{pDataSetIdx});

data_train   = X(:, TrainGroup == 1);
data_test    = X(:, TrainGroup == 0);
labels_train = Labels(:, TrainGroup == 1);
labels_test  = Labels(:, TrainGroup == 0);

whos

opts = struct();
opts.ManifoldDimension = 40;
supMRA = GMRA(data_train, opts, labels_train)


function [X, TrainGroup, Labels] = LoadData(pDataSetName)
% LoadData loads the data and the labels for the classification.
% X: D by N (= Ntrain + Ntest) 
% TrainGroup: 1 by N with 1's corresponding to columns of X to be used as training set.
% Labels: 1 by N of labels

switch pDataSetName
    
    case 'MNIST_Digits_70000'
        
        X = [];
        TrainGroup = [];
        Labels =[];
        
        
    case 'MNIST_Digits_1600'
        
        addpath(genpath('C:\Users\Billy\Desktop\Data'))

        [X, vLabels]=Generate_MNIST([800, 800], struct('Sampling', 'RandN', 'QueryDigits', [0, 1], 'ReturnForm', 'vector')); % n = 9 x p = 784 (=28^2)
        [N, D] = size(X);
        X = X';                         %   X : D by N matrix of N points in D dimensions
        
        Labels = vLabels';              %   Labels      : row N vector of labels for the points
        clear vLabels
        
        % Mix it up (a little bit) to 'see' the classifier result
        temp = X(:,801:1000);
        X(:,801:1000) = X(:, 201:400);
        X(:,201:400) = temp;
        
        temp = X(:,1201:1400);
        X(:,1201:1400) = X(:, 601:800);
        X(:,601:800) = temp;
        
        temp = Labels(:,801:1000);
        Labels(:,801:1000) = Labels(:, 201:400);
        Labels(:,201:400) = temp;
        
        temp = Labels(:,1201:1400);
        Labels(:,1201:1400) = Labels(:, 601:800);
        Labels(:,601:800) = temp;
        
        % Swap again for test data, just to check easily..
        
        temp = X(:,1001:1100);
        X(:,1001:1100) = X(:, 901:1000);
        X(:,901:1000) = temp;
        
        temp = Labels(:,1001:1100);
        Labels(:,1001:1100) = Labels(:, 901:1000);
        Labels(:,901:1000) = temp;
        
        temp = X(:,1401:1500);
        X(:,1401:1500) = X(:, 1301:1400);
        X(:,1301:1400) = temp;
        
        temp = Labels(:,1401:1500);
        Labels(:,1401:1500) = Labels(:, 1301:1400);
        Labels(:,1301:1400) = temp;
        
        clear temp
        
        %          figure; imagesc(Labels); colorbar;
        
        %   TrainGroup  : row N vector, with 1's corresponding to columns of X to be used as training set.
        TrainGroup = zeros(1,N);
        training_idx = 1:N.*1/2;
        TrainGroup(training_idx) = 1;
        
        
    case 'Gaussian_2'
        
        N = 500;
        p = 100; % n = 8;
        mu0 = -3; mu1 = 3;
        sigma0_d = 15; sigma0_c = 0.1;
        sigma1_d = 15; sigma1_c = 0.1;
        %
        % p = 5; % n = 8;
        % mu0 = -3; mu1 = 3;
        % sigma0_d = 0.5; sigma0_c = 0.1;
        % sigma1_d = 0.5; sigma1_c = 0.1;
        
        Mu0 = mu0.*rand(p,1);
        Mu1 = mu1.*rand(p,1);
        
        % Generate two symmetric covariance matrices for two classes
        c = sigma0_c.*rand(p,1);
        Sigma0 = toeplitz(c) + sigma0_d.*eye(p); clear c;
        c = sigma1_c.*rand(p,1);
        Sigma1 = toeplitz(c) + sigma1_d.*eye(p); clear c;
        
        X0 = mvnrnd(Mu0, Sigma0, N)'; % p = 2 x n = 500 matrix
        X1 = mvnrnd(Mu1, Sigma1, N)'; % p = 2 x n = 500 matrix
        X = [X0 X1]; % p = 2 x n = 1000
        Labels =[zeros(1,N) ones(1,N)];
        % N = 100; % 2 groups of iris
        N = 2.*N;
        %   TrainGroup  : row N vector, with 1's corresponding to columns of X to be used as training set.
        TrainGroup = zeros(1,N);
        % training_idx = 1:N.*3/4;
        training_idx = 1:N.*1/2;
        TrainGroup(training_idx) = 1;


    case 'FisherIris'

        load('fisheriris');
        
        speciesNum = nan(size(species));
        for i = 1: numel(species)
            if strcmp(species{i}, 'setosa')
                speciesNum(i) = 0;
            elseif strcmp(species{i}, 'versicolor')
                speciesNum(i) = 1;
            else
                speciesNum(i) = 2;
            end
        end
        
        % N = 150; % 3 groups of iris
        N = 100; % 2 groups of iris
        idx = randperm(N);
        ids = 50;
        % test_idx = idx(1:ids);
        training_idx = idx(ids+1: end);
        
        %% Fitting the parameters for the GMRA_Classifier function.
        %   X           : D by N matrix of N points in D dimensions
        X = meas(1:N,:)';
        %   TrainGroup  : row N vector, with 1's corresponding to columns of X to be used as training set.
        TrainGroup = zeros(1,N);
        TrainGroup(training_idx) = 1;
        %   Labels      : row N vector of labels for the points
        Labels = speciesNum(1:N)';
        
end


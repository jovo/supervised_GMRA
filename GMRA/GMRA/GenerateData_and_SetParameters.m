function [X, GWTopts, imgOpts] = GenerateData_and_SetParameters(pExampleName)

% set GWT parameters
GWTopts = struct();

% The following thresholds are used in the code construct_GMRA.m
GWTopts.threshold1 = sqrt(2)*(1-cos(pi/24));    % threshold of singular values for determining the rank of each ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}
GWTopts.threshold2 = sqrt(2)*sin(pi/24);        % threshold for determining the rank of intersection of ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}

% whether to use best approximations
GWTopts.addTangentialCorrections = false;

% whether to sparsify the scaling functions and wavelet bases
GWTopts.sparsifying = false;
GWTopts.sparsifying_method = 'ksvd'; % or 'spams'

% whether to split the wavelet bases into a common intersection and
% children-specific parts
GWTopts.splitting = false;

% METIS parameters
GWTopts.knn = 30;
GWTopts.knnAutotune = 20;
GWTopts.smallestMetisNet = 10;

% whether to output time
GWTopts.verbose = 1;

% method for shrinking the wavelet coefficients
GWTopts.shrinkage = 'hard';

% whether to avoid using the scaling functions at the leaf nodes and
% instead using the union of their wavelet bases and the scaling functions
% at the parents
GWTopts.avoidLeafnodePhi = false;

% whether to merge the common part of the wavelet subspaces
% associated to the children into the scaling function of the parent.
GWTopts.mergePsiCapIntoPhi  = false;

%% create data, and set additional parameters
imgOpts = struct();

fprintf('\nGenerating/loading %s data...', pExampleName);tic
switch pExampleName
    
    case 'MNIST_Digits'
        
        % generate the dataset
        dataset = struct();
        dataset.N = 5000;
        dataset.digits = 1;
        dataset.projectionDimension = 100;
        
        [X0,GraphDiffOpts,NetsOpts,Labels] = GenerateDataSets( 'BMark_MNIST', ...
            struct('NumberOfPoints',dataset.N,'AutotuneScales',false,'MnistOpts',struct('Sampling', 'FirstN', 'QueryDigits',dataset.digits, 'ReturnForm', 'vector'))); %#ok<ASGLU>
        
        % image parameters
        imgOpts.imageData = true;
        imgOpts.imR = 28;
        imgOpts.imC = 28;
        imgOpts.Labels = Labels;
        
        if dataset.projectionDimension>0 && dataset.projectionDimension<imgOpts.imR*imgOpts.imC,
            imgOpts.X0 = X0;
            imgOpts.cm = mean(X0,2);
            X = X0 - repmat(imgOpts.cm,1,size(X0,2));
            %     [U,S,V] = svd(X,0);
            [U,S,V] = randPCA(X, dataset.projectionDimension);
            %X = U.*repmat(diag(S)', dataset.N, 1);
            X = S*V';
            imgOpts.U = U;
            imgOpts.isCompressed = true;
        else
            X = X0; clear X0;
            imgOpts.isCompressed = false;
        end;
        
        % GWT parameters that need to be set separately
        %GWTopts.ManifoldDimension = 4; % if 0, then determine locally adaptive dimensions using the following fields:
        GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
        GWTopts.errorType = 'relative';
        GWTopts.precision  = .050; % only for leaf nodes
        
    case 'YaleB_Faces'
        
        load YaleB_PCA
        X = S*V'; %#ok<NODEF>
        % image parameters
        imgOpts.imageData = true;
        imgOpts.imR = 480;
        imgOpts.imC = 640;
        
        imgOpts.Labels = Labels; %#ok<NODEF>
        imgOpts.cm =  Imean;
        imgOpts.U = U; %#ok<NODEF>
        imgOpts.isCompressed = true;
        
        % GWT parameters that need to be set separately
        % GWTopts.ManifoldDimension = 4; 
        GWTopts.errorType = 'relative';
        GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
        GWTopts.precision  = 0.05; % only for leaf nodes
        
    case 'croppedYaleB_Faces'
        
        load extendedYaleB_crop_SVD
        dataset.projectionDimension = 500;
        %X = V(:,1:dataset.projectionDimension); %#ok<NODEF>
        X = S(1:dataset.projectionDimension,1:dataset.projectionDimension)*V(:,1:dataset.projectionDimension)'; %#ok<NODEF>
        % image parameters
        imgOpts.imageData = true;
        imgOpts.imR = 192;
        imgOpts.imC = 168;

        %imgOpts.Labels = Labels; %#ok<NODEF>
        imgOpts.cm =  center;
        %imgOpts.V = bsxfun(@times, U(:,1:dataset.projectionDimension), (diag(S(1:dataset.projectionDimension,1:dataset.projectionDimension)))'); %#ok<NODEF>
        imgOpts.U = U(:,1:dataset.projectionDimension); %#ok<NODEF>
        imgOpts.isCompressed = true;
        
        % GWT parameters that need to be set separately
        %GWTopts.ManifoldDimension = 4; 
        GWTopts.threshold0 = .5; % threshold for choosing pca dimension at each nonleaf node
        GWTopts.errorType = 'relative';
        GWTopts.precision  = .05; % only for leaf nodes
        
    case 'SpikeBins'
        
%         load Bin_Spike_Cursor
%         X = binned_parsed;
        
        load Bin_spikes
        X = spikebinned';

%         load Pursuit
%         X = double(spikes)';
        
%         X = X./repmat(sqrt(sum(X.^2,2)), 1, size(X,2));        %#ok<NODEF>
%         X = X';
%         
%         %GWTopts.ManifoldDimension = dataset.k;
%         GWTopts.threshold0 = 0.5;
%         GWTopts.errorType = 'absolute';
%         GWTopts.precision  = 1e-2; % only for leaf nodes
        
    case 'ScienceNews'
        
        load X20
        
        X = X./repmat(sqrt(sum(X.^2,2)), 1, size(X,2));        %#ok<NODEF>
        X = X';
        
        %GWTopts.ManifoldDimension = dataset.k;
        GWTopts.threshold0 = 0.5;
        GWTopts.errorType = 'absolute';
        GWTopts.precision  = 1e-2; % only for leaf nodes
        
    case 'NaturalImagePatches'
        
        load NaturalImagePatches.mat
        X = Y(:, randsample(size(Y,2),10000)); %#ok<NODEF>
        
        % image parameters
        imgOpts.imageData = true;
        imgOpts.imR = 16;
        imgOpts.imC = 16;
        imgOpts.isCompressed = false;
        
        % GWT parameters that need to be set separately
        %GWTopts.ManifoldDimension = 4; 
        GWTopts.errorType = 'relative';
        GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
        GWTopts.precision  = 0.05; % only for leaf nodes
        
    case 'IntegralOperator'
        
        [X0,~,~,Labels]=GenerateDataSets('Planes',struct('NumberOfPoints',5000,'EmbedDim',3,'PlanesOpts',struct('Number',2,'Dim',[1,2],'Origins',zeros(2,3),'NumberOfPoints',[1000,4000],'Dilations',[1,5])));
        X=GenerateIntegralOperatorFromSets(X0(:,Labels==1)', X0(:,Labels==2)');
        
        %GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:
        
        GWTopts.errorType = 'relative';
        GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
        GWTopts.precision  = 0.01; % only for leaf nodes
        
    case 'MeyerStaircase'
        
        % data parameters
        dataset = struct();
        dataset.name = pExampleName;
        
        % % % %         dataset.N = 100; % number of data points to be generated            % This is for Cosine, gives Haar wavelets?
        % % % %         dataset.D = 5000; % ambient dimension of the data
        % % % %         dataset.k = 5000; % intrinsic dimension of the manifold
        
        % % % %         dataset.N = 40; % number of data points to be generated             % Another possibility for Cosine...
        % % % %         dataset.D = 8000; % ambient dimension of the data
        % % % %         dataset.k = 8000; % intrinsic dimension of the manifold
        
        dataset.N = 1000;
        dataset.k = 1;
        dataset.D = 1000;
        dataset.MeyerStepWidth=40;
        dataset.noiseLevel = 0/sqrt(dataset.D);
        
        % Generate data
        X_clean = GenerateDataSets( dataset.name, struct('NumberOfPoints',dataset.N,'Dim',dataset.D,'MeyerStepWidth',dataset.MeyerStepWidth,'EmbedDim',dataset.D,'NoiseType','Gaussian','NoiseParam',0) );
        
        % Add noise
        if dataset.noiseLevel>0,
            X = X_clean + dataset.noiseLevel*random('norm', 0,1, size(X_clean)); 
            GWTopts.X_clean = X_clean;
        else
            X = X_clean;
        end
        
        % GWT parameters that need to be set separately
        GWTopts.ManifoldDimension = dataset.k;
        GWTopts.errorType = 'absolute';
        GWTopts.precision  = 5e-3; % only for leaf nodes     
        
    case 'D-Gaussian' % D-Gaussian
        %% data parameters
        dataset = struct();
        dataset.name = pExampleName;        
        dataset.N = 10000;
        dataset.k = 0;
        dataset.D = 512;
        dataset.noiseLevel = 0.01/sqrt(dataset.D);        
        
        % Generate data
        lFactor = 1/2;
        X_clean = GenerateDataSets( dataset.name, struct('NumberOfPoints',dataset.N,'Dim',dataset.D,'EmbedDim',dataset.D,'NoiseType','Gaussian','NoiseParam',0, ...
            'GaussianMean',[ones(5,1);lFactor^2*ones(10,1);lFactor^3*ones(20,1);lFactor^4*ones(40,1);zeros(dataset.D-75,1)]', ...
            'GaussianStdDev',0.2*[ones(5,1);lFactor^2*ones(10,1);lFactor^3*ones(20,1);lFactor^4*ones(40,1);zeros(dataset.D-75,1)]) ); % figure;plot(idct(X_clean(randi(size(X_clean,1),1),:)))
        
        % Add noise
        if dataset.noiseLevel>0,
            X = X_clean + dataset.noiseLevel*random('norm', 0,1, size(X_clean));
            GWTopts.X_clean = X_clean;
        else
            X = X_clean;
        end
        
        %% GWT parameters that need to be set separately
        GWTopts.ManifoldDimension = 0;
        GWTopts.errorType = 'relative';
        GWTopts.threshold0 = lFactor/2; % threshold for choosing pca dimension at each nonleaf node
        GWTopts.precision  = lFactor/10; % only for leaf nodes        

    otherwise % artificial data
        %% data parameters
        dataset = struct();
        dataset.name = pExampleName;
        
        % % % %         dataset.N = 100; % number of data points to be generated            % This is for Cosine, gives Haar wavelets?
        % % % %         dataset.D = 5000; % ambient dimension of the data
        % % % %         dataset.k = 5000; % intrinsic dimension of the manifold
        
        % % % %         dataset.N = 40; % number of data points to be generated             % Another possibility for Cosine...
        % % % %         dataset.D = 8000; % ambient dimension of the data
        % % % %         dataset.k = 8000; % intrinsic dimension of the manifold
        
        dataset.N = 10000;
        dataset.k = 2;
        dataset.D = 50;
        dataset.noiseLevel = 0;
%         dataset.noiseLevel =  1/sqrt(dataset.D)
        
        % Generate data
        X_clean = GenerateDataSets( dataset.name, struct('NumberOfPoints',dataset.N,'Dim',dataset.k,'EmbedDim',dataset.D,'NoiseType','Uniform','NoiseParam',0) );
        %X_clean = X_clean';         % This is for 'Cosine' only.
        
        % Add noise
        if dataset.noiseLevel>0,
            X = X_clean + dataset.noiseLevel*random('norm', 0,1, size(X_clean)); % N by D data matrix
            GWTopts.X_clean = X_clean;
        else
            X = X_clean;
        end
        
        %% GWT parameters that need to be set separately
        GWTopts.ManifoldDimension = dataset.k;
        %GWTopts.threshold0=0.5;
        GWTopts.errorType = 'absolute';
        GWTopts.precision  = 1e-2; % only for leaf nodes
        
end
fprintf('done. (%.3f sec)',toc);

% threshold for wavelet coefficients
GWTopts.coeffs_threshold = 0; %GWTopts.precision/10;

%figure; do_plot_data(X,[],struct('view', 'pca'));

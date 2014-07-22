% Test LOL for supervised_GMRA with fisheriris data
clear all
close all
clc

dir = fileparts(mfilename('fullpath'));
cd(dir);
addpath(genpath(pwd));

testonFISHERIRIS = 0;
if testonFISHERIRIS
    
    load fisheriris
    
    % Data
    % meas: measurements (n = 150 x d = 4)
    % species of iris: 'setosa', 'versicolor', 'virginica  (150 x 1)
    % Note: some points are overlapping..
    
    %     variable for TRAINING.
    for i = 1: numel(species)
        if strcmp(species{i}, 'setosa')
            labels(i) = 0;
        elseif strcmp(species{i}, 'versicolor')
            labels(i) = 1;
        else
            labels(i) = 2;
        end
    end
    
    N = 100; % 2 groups of iris
    % meas: X: n by d
    X = meas(1:N,:);
    % labels: Y: n by 1
    Y = labels(1:N)';
    clear i labels meas species
    
    idx = randperm(N);
    data = X(idx,:);
    labels = Y(idx, :);
    clear X Y
    
    Ntrain = 30; Ntest = N - Ntrain;
    data_train = data(1:Ntrain, :); % N by d
    data_test = data(Ntrain+1:end, :);
    labels_train = labels(1:Ntrain, :);
    labels_test = labels(Ntrain+1:end, :);
    
    clear idx data labels
    whos
    
    Opts.LOL_alg = 'DENL';
    [ task, ks] = set_task_LOL( Opts, size(data_train,2) )
    
    Opts.task = task;
    Opts.task.ks = ks;
    
    
    [labels_pred_LOL, n_errors_LOL, classifier_LOL, ~] = LOL_traintest( data_train', labels_train, data_test', labels_test, Opts );
    
%     whos
%     classifier_LOL
    for i = 1:length(ks)
        ERR_LOL(i) = sum(labels_pred_LOL(:,i) ~= labels_test);
        data_test_projd{i} = classifier_LOL.Proj{1}.V * data_test';
    end
    
    ACC_LOL = 1 - ERR_LOL./numel(labels_test)
    
    
    Opts.task = task;
    for i = 1:length(ks)
        ks(i)
        Opts.task.ks = ks(i);
        [labels_pred_GMRALOL{i}, n_errors_GMRALOL{i}, classifier_GMRALOL{i}, ~] = LOL_traintest( data_train', labels_train, data_test', labels_test, Opts );
        ERR_GMRALOL(i) = sum(labels_pred_GMRALOL{i} ~= labels_test);
        data_test_projd{i} = classifier_GMRALOL{i}.Proj{1}.V * data_test';
    end
    
    ACC_GMRALOL = 1 - ERR_GMRALOL./numel(labels_test)
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% LoadData

%% Pick a data set

pDataSetNames  = {'MNIST_HardBinary_T60K_t10K', 'MNIST_HardBinary_T5.0K_t5.0K',  'MNIST_HardBinary_T2.5K_t2.5K', 'MNIST_EasyBinary_T2.5K_t2.5K', 'MNIST_EasyBinary_T0.8K_t0.8K', 'MNIST_EasyBinary_T0.7K_t0.7K', 'MNIST_EasyBinary_T0.6K_t0.6K', 'MNIST_EasyBinary_T0.5K_t0.5K', 'MNIST_EasyBinary_T0.4K_t0.4K', 'MNIST_EasyBinary_T0.3K_t0.3K', 'MNIST_EasyBinary_T0.2K_t0.2K', 'MNIST_EasyTriple_T0.6K_t0.6K', 'MNIST_EasyTriple_T0.3K_t0.3K', 'MNIST_EasyTriple_T0.3K_t10K',  'MNIST_EasyAllDigits_T0.3K_t10K', 'MNIST_EasyBinary_T10_t10', 'Gaussian_2', 'FisherIris', 'CIFAR_T10kt10k', 'CIFAR_T50kt10k' };

fprintf('\n Data Sets:\n');
for k = 1:length(pDataSetNames),
    fprintf('\n [%d] %s',k,pDataSetNames{k});
end;
fprintf('\n\n  ');

pDataSetIdx = input('Pick a data set to test: \n');

%% Choose if I want to run LDA or LOL
testLDA = input('(0) Test on LOL only (1)Test on both LDA and LOL (2)Test on LDA only: \n'); % Test on LDA for comparison

if testLDA ~= 2
    %% Choose if I want to use bestK(test all k) or bestTrainK(K trained in CV)
    TrainK = input('TestAllK: 0/ CV_K: 1/ SVDecay_K: 2/ BIC_K:3 (vector for testing multiple methods) \n'); % AllK method is always run.
    
%     %% Choose if I want to test the option of choosing K
%     testK = input('Type 1 to test estimating K: \n');
    
    %% Choose if I want to save the figures
    saveFig = input('Save the figures?: \n')
end

%% How many trials?
nTrials = input('Run how many trials: \n ');

for nT = 1: nTrials
    
    
    %% Load the data set
    
    [X, TrainGroup, Labels] = LoadData(pDataSetNames{pDataSetIdx});
    
    data_train   = X(:, TrainGroup == 1)';
    data_test    = X(:, TrainGroup == 0)';
    labels_train = Labels(:, TrainGroup == 1)';
    labels_test  = Labels(:, TrainGroup == 0)';
    
    if testLDA >0
        tic;
        Timing_onset_LDA = cputime;
        [labels_pred_LDA, n_errors_LDA, ~, ~] = LDA_traintest( data_train', labels_train, data_test', labels_test);
        tictoc_LDA(nT) = toc;
        cputime_LDA(nT) = cputime - Timing_onset_LDA;
        
        
        %        ERR_LDA = NaN(size(labels_pred_LDA,2), 1);
        %        ACC_LDA = NaN(size(labels_pred_LDA,2), 1);
        ERR_LDA = sum(labels_pred_LDA ~= labels_test');
        ACC_LDA(nT) = 1- ERR_LDA./numel(labels_test);
        
        %   if testLDA == 2
        %	return;
        %  end
        
    end
    
    if testLDA ~= 2
        
        Opts.LOL_alg = 'DENL';
        [ task, ks] = set_task_LOL( Opts, size(data_train,2) );
        Opts.task = task;
        Opts.task.ks = ks;
        
        if any(TrainK == 2)
            %% Local dimension based on local singular value decay of PCA and of LOL
            Opts.MaxDim = size(data_train,2);
            if ~isfield(Opts, 'threshold0'),        Opts.threshold0 = 0.1;         end;
            if ~isfield(Opts, 'errorType'),         Opts.errorType = 'relative';   end;
            %             if ~isfield(Opts, 'precision'),         Opts.precision = 0.05;         end;
            if ~isfield(Opts, 'MaxDim'),        Opts.MaxDim = 100;                 end;
            
            % testK with SVD decay
            
            % data_train: N by D
            data_train_mean      = mean(data_train ,2);
            data_train_size      = numel(data_train_mean);
            data_train_centered  = bsxfun(@minus, data_train ,data_train_mean);         % Centered data in the current node
            data_train_radii     = sqrt(max(sum(data_train_centered.^2,1)));
            
            % Compute local SVD
            % Local dimension is not fixed, but based on local singular value decay
            [~,S_PCA,~]             = randPCA(data_train_centered,min([min(size(data_train_centered)), Opts.MaxDim]));    % Use fast randomized PCA
            remEnergy_PCA           = sum(sum(data_train_centered.^2))-sum(diag(S_PCA).^2);
            Sigmas_PCA   = ([diag(S_PCA); sqrt(remEnergy_PCA)]) /sqrt(data_train_size);
            reqDim_PCA(nT) = min(numel(diag(S_PCA)), mindim(Sigmas_PCA, Opts.errorType, Opts.threshold0));
            
            % testK with LOL decay
            
%             types{1} = 'DENL';
%             Kmax = size(data_train, 2);
%             [Proj, P] = LOL(data_train, labels_train, types, Kmax);
%             S_LOL = diag(P.ds);
%             remEnergy_LOL   = sum(sum(data_train_centered.^2))-sum(diag(S_LOL).^2);
%             Sigmas_LOL  = ([diag(S_LOL); sqrt(remEnergy_LOL)]) /sqrt(data_train_size);
%             reqDim_LOL(nT)  = min(numel(diag(S_LOL)), mindim(Sigmas_LOL, Opts.errorType, Opts.threshold0));
            
%             Opts.task.ks(numel(Opts.task.ks)+1) = reqDim_PCA(nT);
        end
            %%
            
        
        if any(TrainK == 1)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%% VALIDATION stage (to choose K with CV) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % CV within training data for estimating k
            
            cputime_LOL_CVK_onset = cputime;
            
            N = size(data_train, 1)
            disp('CV ratio (=Ntest/N): ')
            ratio = 20/100
            disp('ntest: ')
            ntest = floor(ratio*N);
            disp('ntrain: ')
            ntrain = N-ntest
            % 	swp_idx = randperm(N);
            % 	coeffs_swp = coeffs(:,swp_idx);
            % 	dataLabels_swp = dataLabels(swp_idx);
            
            data_test_cv = data_train(1: ntest,:);
            data_train_cv = data_train(ntest+1: end,:);
            labels_test_cv = labels_train(1:ntest);
            labels_train_cv = labels_train(ntest+1:end);
%             whos
            
%             tic;
            [labels_pred_LOL, n_errors_LOL, classifier_LOL, ~] = LOL_traintest( data_train_cv', labels_train_cv, data_test_cv', labels_test_cv, Opts );
%             toc;
%             tic;
            for i = 1:length(ks)
                ERR_LOL(i) = sum(labels_pred_LOL(:,i) ~= labels_test_cv);
                %    data_test_projd{i} = classifier_LOL.Proj{1}.V * data_test';
            end
%             toc;
            ACC_LOL_Val = 1 - ERR_LOL./numel(labels_test_cv);
            [max_ACC_cv, minidx_cv] = max(ACC_LOL_Val);
            
            cputime_LOL_CVK_train(nT) = cputime - cputime_LOL_CVK_onset;
            
            min_k_CV = ks(minidx_cv);
            min_ks_CV(nT) = min_k_CV;
            Opts_CV = Opts;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Opts_CV.task.ks = min_k_CV;
            %%%%%%%%%%%%%%%%%%%%% TRAIN & TEST stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            [labels_pred_LOL_CVK, n_errors_LOL, classifier_LOL, ~] = LOL_traintest( data_train', labels_train, data_test', [], Opts_CV );
            cputime_LOL_CVK(nT) = cputime - cputime_LOL_CVK_onset;
             
            ERR_LOL_onetime = sum(labels_pred_LOL_CVK~= labels_test);
            ACC_LOL_CV(nT) = 1 - ERR_LOL_onetime./numel(labels_test);
            
            
            if any(TrainK==2)
                Opts_CV.task.ks = reqDim_PCA(nT);
                
                [labels_pred_LOL_SVDecayK, n_errors_LOL, classifier_LOL, ~] = LOL_traintest( data_train', labels_train, data_test', [], Opts_CV );
                %             cputime_LOL_SVDK(nT) = cputime - cputime_LOL_CVK_onset;
                
                ERR_LOL_onetime = sum(labels_pred_LOL_SVDecayK~= labels_test);
                ACC_LOL_SVDecayK(nT) = 1 - ERR_LOL_onetime./numel(labels_test);
                
                
            end
            
            % IF WE WANT SEPARATE TRAIN AND TEST STAGE:
            %             %%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             [labels_pred_LOL, n_errors_LOL, classifier_LOL, ~] = LOL_traintest( data_train', labels_train, [], [], Opts_CV );
% 
%             data_test_projd{1} = classifier_LOL.Proj{1}.V * data_test';
% 
%             
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %             tic;
%             for i = 1:1
%                 [n_errors, labels_pred, labels_prob] = LDA_test(classifier_LOL, data_test_projd{i}, labels_test );
%                 ERR_LOL_test(i) = sum(labels_pred ~= labels_test');
%             end
%             cputime_LOL_CVK(nT) = cputime - cputime_LOL_CVK_onset;
% 
%             ACC_LOL_test = 1 - ERR_LOL_test./numel(labels_test);
%             ACC_LOL_CV(nT) = ACC_LOL_test;
        end
        
        %  AllK
%         if any(TrainK == 0) % if BestK
%             
            tic;
            Timing_onset_LOL = cputime;
            [labels_pred_LOL, n_errors_LOL, ~, ~] = LOL_traintest( data_train', labels_train, data_test', labels_test, Opts );
            tictoc_LOL(nT) = toc;
            cputime_LOL(nT) = cputime - Timing_onset_LOL;
            
            
            ERR_LOL = NaN(size(labels_pred_LOL,2), 1);
            % ACC_LOL = NaN(size(labels_pred_LOL,2), nTrials);
            for i = 1: size(labels_pred_LOL,2)
                ERR_LOL(i) = sum(labels_pred_LOL(:,i) ~= labels_test);
            end
            ACC_LOL(:,nT) = 1- ERR_LOL./numel(labels_test);
            
            [maxACC(nT), min_ks_idx(nT)] = max(ACC_LOL(:,nT));
            min_ks = ks(min_ks_idx);
            
            if saveFig
                
                cd('/home/collabor/yb8/supervised_GRMA/TempTest/results/results_LOL/')
                
                h = figure('visible', 'off');
                plot(ks, ACC_LOL, 'x')
                xlabel('ks')
                ylabel('Accuracy')
                title_str = ['LOL-MNIST-T300t10k' num2str(nT)];
                % title(title_str, 'FontSize', 15)
                print(h,'-dpng',title_str)
                
                % set(h, 'ResizeFcn', 'set(gcf,"visible","on")');
                set(h, 'visible', 'on');
                savefig(h, title_str)
                close all;
                
                cd(dir)
            end
%         end
        
    end
end

close all;
figure; plot(ks, mean(ACC_LOL, 2), 'gx-')
hold on; plot(1:max(ks), repmat(mean(maxACC), [1 max(ks)]), 'y-')
hold on; plot(1:max(ks), repmat(mean(ACC_LOL_SVDecayK), [ 1 max(ks)]), 'r-')
hold on; plot(1:max(ks), repmat(mean(ACC_LOL_CV), [1 max(ks)]), 'b-')
legend('LOL for all k', 'maxACC',  'LOL with k from Singular Value Decay', 'LOL with k from CV')
title('MNIST: Training0.3k: Test10k: LabelsAll: #trials=40', 'FontSize', 20)
if testLDA ~= 0
    hold on; plot(1:max(ks), repmat(mean(ACC_LDA), [1 max(ks)]), 'c-')
    legend('LOL for all k', 'maxACC',  'LOL with k from Singular Value Decay', 'LOL with k from CV', 'LDA')
end

% title('CIFAR: Training10k: Test10k: LabelsAll', 'FontSize', 20)
xlabel('k', 'FontSize', 18)
ylabel('Accuracy', 'FontSize', 18)
hold on; plot(reqDim_PCA, ACC_LOL_SVDecayK, 'ro', 'MarkerSize', 11)
hold on; plot(min_k_CV, ACC_LOL_CV, 'bo', 'MarkerSize', 11)


return;






% Test LOL for supervised_GMRA with fisheriris data
clear all
close all
clc

dir = fileparts(mfilename('fullpath'));
cd(dir); 
addpath(genpath(pwd));

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

whos     
classifier_LOL
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


% 
% % Checking whether data_test = [] affects the output of boundary
% [CLASS, ~,~,~,COEF] = classify(data_test(:,1),data_train(:,1),labels_train, 'linear');
% cof = [COEF(1,2).const; COEF(1,2).linear]
% % data_test = [];
% size(data_test)
% size(data_train)
% size(labels_train)
% [CLASS2,~,~,~,COEF2] = classify([],data_train(:,1),labels_train, 'linear');
% cof2 = [COEF2(1,2).const; COEF2(1,2).linear]
% 
% % Testing with one group
% labels_train(:) = 1;
% [CLASS3,~,~,~,COEF3] = classify(data_test,data_train,labels_train, 'linear');
% cof3 = [COEF3(1,2).const; COEF3(1,2).linear]
% %  CLASS3 is output but there is no coef3 if labels_train is unique.
% labels_train(:) = 1;
% [CLASS4,~,~,~,COEF4] = classify([],data_train,labels_train, 'linear');
% cof4 = [COEF4(1,2).const; COEF4(1,2).linear]
% return;
% % => Nope it doesn't!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Let's try the MNIST data here.
% LoadData

%% Pick a data set

pDataSetNames  = {'MNIST_HardBinary_T60K_t10K', 'MNIST_HardBinary_T5.0K_t5.0K',  'MNIST_HardBinary_T2.5K_t2.5K', 'MNIST_EasyBinary_T2.5K_t2.5K', 'MNIST_EasyBinary_T0.8K_t0.8K', 'MNIST_EasyBinary_T0.7K_t0.7K', 'MNIST_EasyBinary_T0.6K_t0.6K', 'MNIST_EasyBinary_T0.5K_t0.5K', 'MNIST_EasyBinary_T0.4K_t0.4K', 'MNIST_EasyBinary_T0.3K_t0.3K', 'MNIST_EasyBinary_T0.2K_t0.2K', 'MNIST_EasyTriple_T0.6K_t0.6K', 'MNIST_EasyTriple_T0.3K_t0.3K', 'MNIST_EasyTriple_T0.3K_t10K',  'MNIST_EasyAllDigits_T0.3K_t10K', 'MNIST_EasyBinary_T10_t10', 'Gaussian_2', 'FisherIris' };
    
fprintf('\n Data Sets:\n');
for k = 1:length(pDataSetNames),
    fprintf('\n [%d] %s',k,pDataSetNames{k});
end;
fprintf('\n\n  ');

pDataSetIdx = input('Pick a data set to test: \n');

%% Load the data set

[X, TrainGroup, Labels] = LoadData(pDataSetNames{pDataSetIdx});

data_train   = X(:, TrainGroup == 1)';
data_test    = X(:, TrainGroup == 0)';
labels_train = Labels(:, TrainGroup == 1)';
labels_test  = Labels(:, TrainGroup == 0)';

%% Choose if I want to use bestK(test all k) or bestTrainK(K trained in CV)
TrainK = input('Type 0 for BestK (test for all ks) and Type 1 for TrainedK (test with k from CV): \n');


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Let's test the LDA first  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % both mauro's LDA and the matlab LDA.
% % matlab LDA could be used to test whether the covariance matrix has a positive definite.
% 
% % data: N by D, labels: N by 1, yet input as d by n and n by 1.
% Opts = [];
% [labels_pred_LDA, n_errors_LDA, classifier_LDA, ~] = LDA_traintest( data_train', labels_train, data_test', labels_test, Opts );
% ERR_LDA = sum(labels_pred_LDA ~= labels_test');
% disp('ACC_LDA:')
% ACC_LDA = 1- ERR_LDA./numel(labels_test)
% % [CLASS] = classify(data_test, data_train, labels_train, 'linear')
% 
% 
% 
% 
% % Let's test the crossval of lda vs. lol %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% cp = cvpartition(labels_train,'k',10);
% opts = statset('UseParallel', 'never');
% % crossval of LDA_traintest %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Opts.Classifier = @LDA_traintest;
% classf = @(xtrain, ytrain,xtest)(Opts.Classifier(xtrain',ytrain',xtest',[],Opts));
% cvMCR = crossval('mcr',data_train ,labels_train ,'predfun', classf,'partition',cp,'Options',opts)
% %           length(dataLabels)
% total_errors   = cvMCR*length(labels_train);
% 
% 
% % crossval of LOL_traintest %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Opts.LOL_alg = 'DENL';
% [ task, ks] = set_task_LOL( Opts, size(data_train,2) );
% ks
% 
% Opts.task = task;
% Opts.Classifier = @LOL_traintest;
% 
% for i = 1:length(ks)
% %         disp('displaying the k')
% %          ks(i)
% %	    disp('cross val start of loop')
%             Opts.task.ks = ks(i);
%             classf = @(xtrain, ytrain,xtest)(Opts.Classifier(xtrain',ytrain',xtest',[],Opts));
%             cvMCR(i) = crossval('mcr',data_train ,labels_train ,'predfun', classf,'partition',cp,'Options',opts)
% %           length(dataLabels)
% 	 total_errors_ks(i)    = cvMCR(i)*length(labels_train);
% end
%  


% labels_test = [];
% [labels_pred_LDA2, n_errors_LDA2, classifier_LDA2, ~] = LDA_traintest( data_train', labels_train, data_test', labels_test, Opts );

% 
% % Testing the LOL now %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % LOL_traintest input: data: D by N, labels: N by 1  
% Opts.LOL_alg = 'DENL';
% [ task, ks] = set_task_LOL( Opts, size(data_train,2) );
% Opts.task = task;
% Opts.task.ks = ks;
%  
% tic;
% [labels_pred_LOL, n_errors_LOL, classifier_LOL, ~] = LOL_traintest( data_train', labels_train, data_test', labels_test, Opts );
% toc;
% tic;
% for i = 1:length(ks)
%    ERR_LOL(i) = sum(labels_pred_LOL(:,i) ~= labels_test);
%     data_test_projd{i} = classifier_LOL.Proj{1}.V * data_test';
% end
% toc;
% ACC_LOL = 1 - ERR_LOL./numel(labels_test) 
% 
whos

% Separate Validation, Train, Test stage for estimation of optimal k %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% labels_test = [];
% data_test = [];

Opts.LOL_alg = 'DENL';
[ task, ks] = set_task_LOL( Opts, size(data_train,2) );
Opts.task = task;
Opts.task.ks = ks;

if TrainK

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Validation stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CV within training data for estimating k
    
    N = size(data_train, 1)
    disp('CV ratio (=Ntest/N): ')
    ratio = 90/100
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
    whos
   
    tic;
    [labels_pred_LOL, n_errors_LOL, classifier_LOL, ~] = LOL_traintest( data_train_cv', labels_train_cv, data_test_cv', labels_test_cv, Opts );
    toc;
    tic;
    for i = 1:length(ks)
        ERR_LOL(i) = sum(labels_pred_LOL(:,i) ~= labels_test_cv);
        %    data_test_projd{i} = classifier_LOL.Proj{1}.V * data_test';
    end
    toc;
    ACC_LOL = 1 - ERR_LOL./numel(labels_test_cv);
    [max_ACC_cv, minidx_cv] = max(ACC_LOL);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Opts.task.ks = ks(minidx_cv);
    disp('mink: ')
    ks(minidx_cv)
    tic;
    [labels_pred_LOL, n_errors_LOL, classifier_LOL, ~] = LOL_traintest( data_train', labels_train, [], [], Opts );
    toc;
    disp('classifier: ')
    classifier_LOL
    tic;
    data_test_projd{1} = classifier_LOL.Proj{1}.V * data_test';
    toc;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST stage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    for i = 1:1
        %  [n_errors, labels_pred, labels_prob] = LOL_test(classifier_GMRALOL{i}, data_test_projd{i}, labels_test );
        [n_errors, labels_pred, labels_prob] = LDA_test(classifier_LOL, data_test_projd{i}, labels_test );
        ERR_LOL_test(i) = sum(labels_pred ~= labels_test');
    end
    toc;
    ERR_LOL_test
    
    ACC_LOL_test = 1 - ERR_LOL_test./numel(labels_test);
    max(ACC_LOL_test)
    min(ACC_LOL_test)
    
else % if BestK
    
    [labels_pred_LOL, n_errors_LOL, ~, ~] = LOL_traintest( data_train', labels_train, data_test', labels_test, Opts );
    
    ERR_LOL = NaN(size(labels_pred_LOL,2), 1);
    ACC_LOL = NaN(size(labels_pred_LOL,2), 1);
    for i = 1: size(labels_pred_LOL,2)
        ERR_LOL(i) = sum(labels_pred_LOL(:,i) ~= labels_test);
    end
    ACC_LOL = 1- ERR_LOL./numel(labels_test);
    %
    %     f = figure('visible', 'off');
    %     plot(ks, ACC_LOL, 'x');
    %     print(f, 'r-80', 'dpng', 'image1.png')
    %     xlabel('ks'); ylabel('Accuracy')
    %     title('Accuracy: Alg = LOL, Data = MNIST:T0.3k,t10k')
    %
    %
    
    
    %     f = figure('visible', 'off')
    %     plot(ks,ACC_LOL, 'x')
    %     title_str = ['Accuracy: Alg = LOL, Data = MNIST:T0.3k,t10k'];
    %     set(gcf,'NextPlot','add');
    %     axes;
    %     h = title(title_str, 'FontSize', 15);
    %     set(gca,'Visible','off');
    %     set(h,'Visible','on');
    %     set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 30 20])
    %     saveas(h, title_str, 'png');
    
    h = figure('visible', 'off');
    plot(ks, ACC_LOL, 'x')
    
    print(h,'-dpng','TEST.png')
    
    
end

return;












for i = 1:length(ks)
%    	disp('this time the k is: ')
%	ks(i)
	Opts.task.ks = ks(i);
    ERR_LOL(i) = sum(labels_pred_LOL{i} ~= labels_test);
    data_test_projd{i} = classifier_LOL{i}.Proj{1}.V * data_test';
end

ACC_LOL = 1 - ERR_LOL./numel(labels_test) ;
disp('max and min of ACC_LOL')
max(ACC_LOL)
min(ACC_LOL)
% classifier_LOL{1}
% classifier_LOL{end}

return;
% LEt's test for classify_cv and classify_train
disp('Testing the CV CV CV CV CV CV CV CV CV CV CV CV CV CV CV CV CV CV CV CV CV')
Opts.LOL_alg = 'DENL';
[ task, ks] = set_task_LOL( Opts, size(data_train,2) );
ks

data_test_CV = [];
labels_test_CV = [];
Opts.task = task;
for i = 1:length(ks)
    	disp('this time the k is: ')
	ks(i)
	Opts.task.ks = ks(i);
    [labels_pred_GMRALOL_CV{i}, n_errors_GMRALOL{i}, classifier_GMRALOL{i}, ~] = LOL_traintest( data_train', labels_train, data_test_CV', labels_test_CV, Opts );

%    ERR_GMRALOL(i) = sum(labels_pred_GMRALOL_CV{i} ~= labels_test);
    data_test_projd{i} = classifier_GMRALOL{i}.Proj{1}.V * data_test';
end
labels_pred_GMRALOL_CV
% ACC_GMRALOL_CV = 1 - ERR_GMRALOL./numel(labels_test) ;
% max(ACC_GMRALOL)
% min(ACC_GMRALOL)
% classifier_GMRALOL{1}
% classifier_GMRALOL{end}




disp('TEST stage TEST stage TEST stage TEST stage TEST stage TEST stage TEST stage')


% n_errors_GMRALOL
% [minerr, minkidx] = min(ERR_GMRALOL)
% ks
% mink = ks(minkidx)
% classifier_GMRALOL
% classifier_GMRALOL{minkidx}
% size(classifier_GMRALOL{minkidx}.Proj{1}.V)
% size(data_test)

% data_test_projd = classifier_GMRALOL{minkidx}.Proj{1}.V *  data_test';

for i = 1:length(ks)
  %  [n_errors, labels_pred, labels_prob] = LOL_test(classifier_GMRALOL{i}, data_test_projd{i}, labels_test );
    [n_errors, labels_pred, labels_prob] = LDA_test(classifier_GMRALOL{i}, data_test_projd{i}, labels_test );
    ERR_GMRALOL_test(i) = sum(labels_pred ~= labels_test');
end
ERR_GMRALOL_test

ACC_GMRALOL_test = 1 - ERR_GMRALOL_test./numel(labels_test) ;
max(ACC_GMRALOL_test)
min(ACC_GMRALOL_test)

return; 

% LOL_traintest input: data: D by N, labels: N by 1  
Opts.LOL_alg = 'DENL';
[ task, ks] = set_task_LOL( Opts, size(data_train,2) );
Opts.task = task;
for i = 1:length(ks)
    Opts.task.ks = ks(i);
    [labels_pred_GMRALOL{i}, n_errors_GMRALOL{i}, classifier_GMRALOL{i}, ~] = LOL_traintest( data_train', labels_train, data_test', labels_test, Opts );
    ERR_GMRALOL{i} = sum(labels_pred_GMRALOL{i} ~= labels_test)
end






% Opts.LOL_alg = {'NNNL'};
% [ task, ks] = set_task_LOL( Opts, size(data_train,2) )
% Opts.task = task;
% [labels_pred_GMRALDA, n_errors_GMRALDA, classifier_GMRALDA, ~] = LOL_traintest( data_train, labels_train, data_test, labels_test, Opts );

% classifier_LDA.W(1,:) - classifier_LDA.W(2,:) == classifier_GMRALDA.W{3}
 
 
% Let's test the LDA_train vs LOL_train


% 
% Input = [ 1 2 3 7 8 9; 12 11 10 15 16 17 ]';% n = 6 by d = 2
% Target = [1 1 1 2 2 2]'; % n = 6 by d = 1
% [classifier_trained] = LDA_train(Input,Target); % ouput a struct
% classifier_trained.W % 2 by 3
% classifier_trained.ClassLabel % 2 by 1
% 
% [CLASS,ERR,POSTERIOR,LOGP,coef] = classify(Input,Input,Target,'linear'); 

% coef.type
% coef.name1
% coef.name2
% coef.const
% coef.linear
% 
% A = [coef(1,2).const; coef(1,2).linear]

% % Test Linear Discriminant Analysis (LDA)
% 
% % % My own small example
% % % n1 x d sample matrix, n2 x d training matrix, 1 x n group matrix
% % training = [ 1:20 51:70;
% %     101:120 151:170]'; % n x d
% % sample = training; % n x d
% % group = [zeros(1,20) ones(1,20)]; % 1 x n
% % class = classify(sample, training, group)
% % % Gives an error that pooled covariance must be  a positive definite.
% % % Doesn't work because observations are linear combination of another.
% % 
% % % Ok, another example of mine.
% % mu1 = [1 -1]; Sigma1 = [.9 .4; .4 .3];
% % X1 = mvnrnd(mu1, Sigma1, 500)'; % p = 2 x n = 500 matrix 
% % 
% % mu2 = [3 1]; Sigma2 = [.9 .4; .4 .3];
% % X2 = mvnrnd(mu2, Sigma2, 500)'; % p = 2 x n = 500 matrix 
% % figure; plot(X1(1,:), X1(2,:), 'x'); hold on;
% % plot(X2(1,:), X2(2,:), 'rx');
% % 
% % training = [X1 X2]' % n = 1000 x p = 2 matrix
% % sample = training;
% % group = [zeros(1, 500) ones(1, 500)];
% % class = classify(sample, training, group); 
% % 
% % 
% 
% % Famous Fisher Iris example
% load fisheriris;
% 
% % Data 
% % meas: measurements (n = 150 x d = 4)
% % species of iris (º×²É): 'setosa', 'versicolor', 'virginica  (150 x 1)
% % Note: some points are overlapping..
% 
% figure; plot(meas(:,3:4));
% figure; scatter(meas(:,3), meas(:,4))
% 
% % fitcdiscr MATLAB 2014a
% % classify MATLAB 
% % CLASS = classify(SAMPLE,TRAINING,GROUP) classifies each row of the data
% %     in SAMPLE into one of the groups in TRAINING.  SAMPLE and TRAINING must
% %     be matrices with the same number of columns.  GROUP is a grouping
% %     variable for TRAINING. 
% for i = 1: numel(species)
%     if strcmp(species{i}, 'setosa')
%         speciesNum(i) = 0;
%     elseif strcmp(species{i}, 'versicolor')
%         speciesNum(i) = 1;
%     else
%         speciesNum(i) = 2;
%     end
% end
% 
% % CV(Cross-Validation) index
% % n1 x d sample matrix, n2 x d training matrix, 1 x n group matrix
% % total_idx = 1: 150; % 1 x 150 
% % test_idx = randsample(150, 50)'; % 1x 50
% % trainig_idx = setdiff(total_idx, test_idx);
% % Note: setdiff acts wieeeerddddd....! Let's just use randperm instead..
% 
% N = 150; % 3 groups of iris
% N = 100; % 2 groups of iris
% idx = randperm(N);
% ids = 50;
% test_idx = idx(1:ids);
% training_idx = idx(ids+1: end);
% 
% training = meas(training_idx, 3:4);
% sample = meas(test_idx, 3:4);
% group_test = speciesNum(test_idx);
% group_train = speciesNum(training_idx);
% 
% % Scatter for training_idx
% SL_test =  meas(test_idx,3);
% SW_test =  meas(test_idx,4);
% SL_train = meas(training_idx,3);
% SW_train = meas(training_idx,4);
% 
% figure;
% h1 = gscatter(SL_train,SW_train,group_train,'rbg','vvv',[],'off');
% set(h1,'LineWidth',2)
% legend('Fisher versicolor','Fisher virginica',...
%        'Location','NW')
%    hold on;
% h2 = gscatter(SL_test,SW_test,group_test,'mmm','xxx',[],'off');
% set(h2,'LineWidth',2)
% legend('Fisher versicolor','Fisher virginica',...
%        'Location','NW')
%    
% figure; 
% scatter(SL_test, SW_test, 'x');
% title('test data')
% 
% %% LDA Method 1: classify.m (MATLAB built-in)
% % % Classify a grid of measurements on the same scale.
% % [X,Y] = meshgrid(linspace(4.5,8),linspace(2,4));
% % X = X(:); Y = Y(:);
% % [C,err,P,logp,coeff] = classify([X Y],[SL SW],...
% %     group,'Quadratic');
%  
% %     [CLASS,ERR,POSTERIOR,LOGP,COEF] = classify(...) returns COEF, a
% %     structure array containing coefficients describing the boundary between
% %     the regions separating each pair of groups.  Each element COEF(I,J)
% %     contains information for comparing group I to group J, defined using
% %     the following fields:
% %         'type'      type of discriminant function, from TYPE input
% %         'name1'     name of first group of pair (group I)
% %         'name2'     name of second group of pair (group J)
% %         'const'     constant term of boundary equation (K)
% %         'linear'    coefficients of linear term of boundary equation (L)
% %         'quadratic' coefficient matrix of quadratic terms (Q)
% %  
% [linclass, err, post, logp, str] = classify(sample, training, group_train);
% 
% figure;
% gscatter(SL_test, SW_test, linclass)
% % hold on; gscatter(X,Y,C,'rb','.',1,'off'); hold off
% 
% % Draw boundary between two regions
% hold on
% K1 = str(1,2).const;
% L1 = str(1,2).linear;
% % Plot the curve K + [x,y]*L + [x,y]*Q*[x,y]' = 0:
% f1 = @(x,y) K1 + L1(1)*x + L1(2)*y
% ezplot(f1);
% % 
% % hold on
% % K2 = str(2,3).const;
% % L2 = str(2,3).linear;
% % % Plot the curve K + [x,y]*L + [x,y]*Q*[x,y]' = 0:
% % f2 = @(x,y) K2 + L2(1)*x + L2(2)*y
% % ezplot(f2);
% % 
% % hold on
% % K3 = str(1,3).const;
% % L3 = str(1,3).linear;
% % % Plot the curve K + [x,y]*L + [x,y]*Q*[x,y]' = 0:
% % f3 = @(x,y) K3 + L3(1)*x + L3(2)*y
% % ezplot(f3);
% 
% % True labels
% hold on;
% gscatter(SL_test, SW_test, group_test, 'rb', 'o')
% % hold on; gscatter(X,Y,C,'rb','.',1,'off'); 
% 
% hold off
% title('Classification of Fisher iris data with MATLAB built-in LDA')
% 
% % %% LDA Method 2: LDA_train_and_predict.m (jovo's)
% % Jovo's LDA_train
% addpath(genpath('C:\Users\Billy\Documents\GitHub\LOL'))
% 
% Xtrain = training'; %  D = 2 x n = 50
% Ytrain = group_train; % 1x n = 50
% Xtest = sample'; % D = 2 x n = 50
% 
% % Phat = LDA_train(X,Y);
% [Yhat, eta, parms] = LDA_train_and_predict(Xtrain, Ytrain, Xtest)
% % % INPUT:
% % %   Xtrain in R^{D x n}: predictor matrix
% % %   Ytrain in {0,1,NaN}^n: predictee matrix
% % %   Xtest in R^{D x n}: data matrix where columns are data points
% % % 
% % % OUTPUT: Phat, a structure of parameters 
% % % Output:
% % %   Yhat in {0,1}^n: vector of predictions
% % %   eta in R^{n x 1}: vector of magnitudes
% % 
% % 
% % % Comparison of MATLAB-built-in classify.m and jovo's LDA code
% % whos linclass Yhat
% % find(Yhat - linclass ~= 0)
% % 

%% LDA Method 3: Mauro's LDA vs. classify
clear all
N = 500;
p = 10; % n = 8;
mu0 = -2.5; mu1 = 2.5;
sigma0_d = 0.5; sigma0_c = 0.1;
sigma1_d = 0.5; sigma1_c = 0.1;


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
Labels =[zeros(N,1); ones(N,1)];
% N = 100; % 2 groups of iris

figure; gscatter(X(1,:), X(2,:), Labels)
title('True Labels for training+test X')

idx = randperm(2*N);
ids = 200;
test_idx = idx(1:ids);
training_idx = idx(ids+1: end);

training = X(:, training_idx)';
sample = X(:, test_idx)';
group_test = Labels(test_idx);
group_train = Labels(training_idx);

% mauro's vs. matlab's vs. jovo's for POSTERIROR AND CLASSIFIER
[labels_pred_matlab, n_errors_matlab, labels_prob_matlab, ~, classifier_matlab] = classify(sample, training, group_train, 'linear', 'empirical');
% sample: N by d, training: N by d, group_train: d by N
% [labels_pred_mauro, n_errors, classifier_mauro, labels_prob] = LDA_traintest( data_train, labels_train, data_test, labels_test, Opts )

rmpath(genpath('C:\Users\Billy\Documents\GitHub\LOL'))
[labels_pred_mauro, n_errors, classifier_mauro, labels_prob] = LDA_traintest( training', group_train, sample', []);

addpath(genpath('C:\Users\Billy\Documents\GitHub\LOL'))
[labels_pred_jovo, eta, P] = LDA_train_and_predict( training', group_train, sample');

%% Comparison with matlab vs. mauro's

% whos labels_pred_matlab labels_pred_mauro labels_pred_jovo
find(labels_pred_matlab ~= labels_pred_mauro')
% find(labels_pred_matlab ~= labels_pred_jovo)

% Comparison of Posterior Prob vector
whos labels_prob_matlab labels_prob eta
dummy(:,1) = labels_prob_matlab(:,2);
% dummy(:,3) = labels_prob;

% Comparison of Classifier coefficients
% L = P.del'*P.InvSig;
% C = - P.del'*P.InvSig*P.mu - P.thresh;
% L
% classifier_matlab.linear
% C
% classifier_matlab.const



%% Comparison with matlab's and jovo's
% Comparison of Predicted Label
whos labels_pred_matlab labels_pred_mauro labels_pred_jovo
find(labels_pred_matlab ~= labels_pred_mauro')
find(labels_pred_matlab ~= labels_pred_jovo)

% Comparison of Posterior Prob vector
whos labels_prob_matlab labels_prob eta
dummy(:,1) = labels_prob_matlab(:,2);
% dummy(:,3) = labels_prob;

% A = [ 1 2; 5 3 ; 4 6; 8 7];
% result = exp(bsxfun(@minus, A, max(A,[],2)))
% B = exp(-1.*abs(A(:,1)- A(:,2)))
% C = ones(size(result));
% C(find(A(:,1)- A(:,2) < 0), 1) = B(find(A(:,1)- A(:,2) < 0))
% C(find(A(:,1)- A(:,2) > 0), 2) = B(find(A(:,1)- A(:,2) > 0))

eta = eta';
B = exp(-1.*abs(eta));
C = ones(size(repmat(eta,1,2)));
C(find(eta < 0), 1) = B(find(eta < 0));
C(find(eta > 0), 2) = B(find(eta > 0));
eta = C;
labels_prob_jovo = eta;
whos labels_prob_matlab eta
find(labels_prob_matlab ~= eta)
figure; plot(labels_prob_matlab(1:10), 'gx')
hold on; plot(eta(1:10), 'ro')

figure; imagesc(labels_prob_matlab); colorbar;
figure; imagesc(eta); colorbar;

diff = abs(labels_prob_matlab - eta);
figure; imagesc(diff); colorbar;
max(max(diff))

% Comparison of Classifier coefficients
L = P.del'*P.InvSig;
C = - P.del'*P.InvSig*P.mu - P.thresh;
classifier_jovo.linear = L;
classifier_jovo.const = C;
L
classifier_matlab.linear
C
classifier_matlab.const

figure; gscatter(sample(:,1), sample(:,2), group_test);
title('True Label for test data')

% Plot with boundary from matlab's.
% figure; plot(X(1,:), X(2,:), 'x')
figure; gscatter(sample(:,1), sample(:,2), labels_pred_matlab)
title('Matlab LDA')
% Draw boundary between two regions
hold on
K1 = classifier_matlab(1,2).const;
L1 = classifier_matlab(1,2).linear;
% Plot the curve K + [x,y]*L + [x,y]*Q*[x,y]' = 0:
f1 = @(x,y) K1 + L1(1)*x + L1(2)*y
ezplot(f1);

% Plot with boundary from jovo's.
% figure; plot(X(1,:), X(2,:), 'x')
figure; gscatter(sample(:,1), sample(:,2), labels_pred_jovo)
title('Jovos LDA')
% Draw boundary between two regions
hold on
K2 = C;
L2 = L;
% Plot the curve K + [x,y]*L + [x,y]*Q*[x,y]' = 0:
f2 = @(x,y) K2 + L2(1)*x + L2(2)*y
ezplot(f2);

% Plot with boundary from mauro's.
% figure; plot(X(1,:), X(2,:), 'x')
figure; gscatter(sample(:,1), sample(:,2), labels_pred_mauro)
title('Mauros LDA')
% Draw boundary between two regions
hold on
K3 = classifier_mauro.W(1,1)-classifier_mauro.W(2,1);
L3 = classifier_mauro.W(1,2:end) - classifier_mauro.W(2,2:end);
% Plot the curve K + [x,y]*L + [x,y]*Q*[x,y]' = 0:
f3 = @(x,y) K3 + L3(1)*x + L3(2)*y
ezplot(f3);


%%% Comparison result: same to nearest tenth. 


% %% LDA Method 4: Bytefish.de/ with one-by-one steps
% 
% X = [2 3;3 4;4 5;5 6;5 7;2 1;3 2;4 2;4 3;6 4;7 6]; % n= 11 x d = 2
% c = [  1;  1;  1;  1;  1;  2;  2;  2;  2;  2;  2]; % n= 11 x d = 1
% 
% c1 = X(find(c==1),:); % n = 5 by d = 2
% c2 = X(find(c==2),:); % n = 6 by d = 2
% 
% %%%
% figure;
% p1 = plot(c1(:,1), c1(:,2), 'ro', 'markersize',10, 'linewidth', 3); hold on;
% p2 = plot(c2(:,1), c2(:,2), 'go', 'markersize',10, 'linewidth', 3)
% xlim([0 8])
% ylim([0 8])
% 
% classes = max(c)
% mu_total = mean(X)
% mu = [ mean(c1); mean(c2) ]
% Sw = (X - mu(c,:))'*(X - mu(c,:))
% Sb = (ones(classes,1) * mu_total - mu)' * (ones(classes,1) * mu_total - mu)
% 
% [V, D] = eig(Sw\Sb)
% 
% % sort eigenvectors desc
% [D, i] = sort(diag(D), 'descend');
% V = V(:,i);
% 
% scale = 5
% pc1 = line([mu_total(1) - scale * V(1,1) mu_total(1) + scale * V(1,1)], [mu_total(2) - scale * V(2,1) mu_total(2) + scale * V(2,1)]);
% 
% set(pc1, 'color', [1 0 0], 'linestyle', '--')
% 
% 
% Xm = bsxfun(@minus, X, mu_total)
% z = Xm*V(:,1)
% % and reconstruct it
% p = z*V(:,1)'
% p = bsxfun(@plus, p, mu_total)
% 
% % delete old plots
% delete(p1);delete(p2);
% 
% y1 = p(find(c==1),:)
% y2 = p(find(c==2),:)
% 
% p1 = plot(y1(:,1),y1(:,2),'ro', 'markersize', 10, 'linewidth', 3);
% p2 = plot(y2(:,1), y2(:,2),'go', 'markersize', 10, 'linewidth', 3);
% 
% 


% %% Now onto jovos' LOL code
% 
% [Yhat_LOL, eta, Proj] = LOL_train_and_predict(Xtrain, Ytrain, Xtest, 1)
% % [Yhat_LOL, eta, Proj] = LOL_train_and_predict(Xtrain, Ytrain, Xtest, 2)
% % train Low Rank Linear Discriminant Analysis Classifier
% 
% % INPUT:
% %   X in R^{D x n}: predictor matrix
% %   Y in R^n: predictee vector
% %   varargin: two options
% %   if nargin == 1, then 
% %       k: projection matrix dimension
% %   else nargin == 2
% %       delta in R^D: mu_0 - mu_1
% %       V in R^{d x D}: projection matrix
% %   end
% % 
% % OUTPUT: Proj in R^{d x D}: projection matrix
% 
% % Comparison of classify.m, jovo's LDA code, and LOL code
% whos Yhat_LOL Yhat linclass
% find(Yhat - linclass ~= 0)
% find(Yhat - Yhat_LOL ~= 0)
% 
% figure;
% gscatter(SL_test, SW_test, Yhat_LOL)
% % hold on; gscatter(X,Y,C,'rb','.',1,'off'); hold off
% 
% % Draw boundary between two regions
% hold on
% K1 = str(1,2).const;
% L1 = str(1,2).linear;
% % Plot the curve K + [x,y]*L + [x,y]*Q*[x,y]' = 0:
% f1 = @(x,y) K1 + L1(1)*x + L1(2)*y
% ezplot(f1);
% 
% % True labels
% hold on;
% gscatter(SL_test, SW_test, group_test, 'rb', 'o')
% % hold on; gscatter(X,Y,C,'rb','.',1,'off'); 
% 
% hold off
% title('Classification of Fisher iris data with LOL')
% 
% 
% % Reading MNIST data
% addpath(genpath('C:\Users\Billy\Desktop\Data\MNIST'))
% % images = loadMNISTImages('train-images.idx3-ubyte');
% load('TrainImages')
% load('TrainImageLabels')
% % [X,vLabels]=Generate_MNIST(NofPts, mOpts) 
% addpath(genpath('C:\Users\Billy\Desktop\GMRA\DiffusionGeometry\Examples'))
% % [X, vLabels]=Generate_MNIST([3, 3, 3], struct('Sampling', 'RandN', 'QueryDigits', [1, 6, 9], 'ReturnForm', 'vector')); % n = 9 x p = 784 (=28^2)
% % IN: 
% %    NofPts: a number or a vector, the number of points to extract. The vector case, the number of points from each digits. See QueryPts in the Opts.
% %    [mOpts]: structure containing the following fields:
% %             [Sampling]: 'FirstN' returns the first NofPts, 'RandN' returns the NofPts points after a random permutation. Default = RandN
% %             [QueryDigits]: if it is empty, it doesn't distinguish digits. If it is a vector cosisting of digits, only those digits are sampled. 
% %                         ex) [0, 3, 5]. In this case, the NofPts should be matched. Otherwise, it samples NofPts(1) points from each digit.
% %                         Default =[].
% %             [ReturnForm]: if 'vector', it returns a point as a 28^2 x 1 vector. if 'matrix', it returns as a 28x28 matrix. Defalut = 'vector' 
% %
% % OUT: a double sum(NofPts)x28^2 matrix if 'vector' is the ReturnForm, a double sum(NofPts)x28x28 3-dim array if 'matrix'. 
% %
% % Example:  X=Generate_MNIST([500, 500, 500], struct('Sampling', 'RandN', 'QueryDigits', [1, 6, 9], 'ReturnForm', 'matrix'));
% %           X=Generate_MNIST(1000);
% 
% % Confirm TrainImages
% % A = squeeze(im2double(TrainImages(n,:,:)));
% 
% % Confirm X, the result of Generate_MNIST
% % for i = 1: 9
% %     A = reshape(X(i,:), 28, 28);
% %     figure; imshow(A);
% % end
% 
% % LDA and LOL the MNIST data for less than 5 and the other
% 
% % [X, vLabels]=Generate_MNIST(100.*ones(1,10), struct('Sampling', 'RandN', 'QueryDigits', 0:9, 'ReturnForm', 'vector')); % Xtrain: n = 1000 x p = 784/ Ytrain: n = 1000 x p = 1
% [X, vLabels]=Generate_MNIST(100.*ones(1,10), struct('Sampling', 'RandN', 'QueryDigits', 0:9, 'ReturnForm', 'vector')); % Xtrain: n = 1000 x p = 784/ Ytrain: n = 1000 x p = 1
% vLabels(vLabels<5) = 0; vLabels(vLabels>=5) = 1;
% % [Xtrain, Ytrain]=Generate_MNIST(50.*ones(1,10), struct('Sampling', 'RandN', 'QueryDigits', 0:9, 'ReturnForm', 'vector')); % Xtrain: n = 500 x p = 784/ Ytrain: n = 500 x p = 1
% 
% [N, D] = size(X)
% idx = randperm(N);
% ids = 250;
% test_idx = idx(1:ids)
% training_idx = idx(ids+1: end);
% 
% Xtrain = X(training_idx, :)'; % d = 784 x n = 250
% Ytrain = vLabels(training_idx,:)'; % d = 1 x n = 250
% Xtest = X(test_idx, :)'; % d = 784 x n = 250
% Ytest = vLabels(test_idx,:)'; % d = 1 x n = 250
% 
% addpath(genpath('C:\Users\Billy\Documents\GitHub\LOL'))
% 
% tic;
% [Yhat_LDA, eta_LDA, parms_LDA] = LDA_train_and_predict(Xtrain, Ytrain, Xtest);
% timing_LDA = toc;
% ratio_LDA = 1 - sum(Ytest~=Yhat_LDA')/(sum(Ytest==Yhat_LDA') + sum(Ytest~=Yhat_LDA'))
% 
% count = 1;
% for k = 1: 100: 780
% tic;
% [Yhat_LOL, eta_LOL, Proj_LOL] = LOL_train_and_predict(Xtrain, Ytrain, Xtest, k);
% timing_LOL(count) = toc;
% ratio_LOL(count) = 1 - sum(Ytest~=Yhat_LOL')/(sum(Ytest==Yhat_LOL') + sum(Ytest~=Yhat_LOL'));
% count = count + 1;
% end
% k = 1: 100: 780;
% figure; plot(k, ratio_LOL)
% 
% % Comparison of classify.m, jovo's LDA code, and LOL code
% % whos Yhat_LOL Yhat linclass
% % find(Yhat - linclass ~= 0)
% % find(Yhat - Yhat_LOL ~= 0)
% 
% 
% % QOL & QOQ
% 
% 
% Phat = QDA_train(Xtrain, Ytrain);
% 
% count = 1;
% for k = 1: 100: 780
% tic;
% [Yhat_QOL, eta_QOL, Proj_QOL] = QOL_train_and_predict(Xtrain, Ytrain, Xtest, Phat.del, k);
% timing_QOL(count) = toc;
% ratio_QOL(count) = 1 - sum(Ytest~=Yhat_QOL')/(sum(Ytest==Yhat_QOL') + sum(Ytest~=Yhat_QOL'));
% count = count + 1;
% end
% k = 1: 100: 780;
% figure; plot(k, ratio_LOL)
% 
% 
% 
% count = 1;
% for k = 1: 100: 780
% tic;
% [Yhat_QOQ, eta_QOQ, Proj_QOQ] = QOQ_train_and_predict(Xtrain, Ytrain, Xtest, Phat.del, k);
% timing_QOQ(count) = toc;
% ratio_QOQ(count) = 1 - sum(Ytest~=Yhat_LOL')/(sum(Ytest==Yhat_LOL') + sum(Ytest~=Yhat_LOL'));
% count = count + 1;
% end
% k = 1: 100: 780;
% figure; plot(k, ratio_LOL)

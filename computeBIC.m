function BIC = computeBIC(X_train, Labels_train, X_test, paramsBIC, classifier)
%computeBIC.m computes the BIC value of each LOL model with different k.

%% Parameters for BIC

% disp('PriorProb, PooleCov, invCov, GroupMean, Group, nGroup')
%   size(paramsBIC.PriorProb)
%   size(paramsBIC.PooledCov)
%   size(paramsBIC.invCov)
%   size(paramsBIC.GroupMean)
%   A = paramsBIC.PriorProb;
%   B = paramsBIC.PooledCov;
%   C = paramsBIC.invCov;
%   D = paramsBIC.GroupMean;
%   E = paramsBIC.Group;
%   F = paramsBIC.nGroup;
% whos A B C D E F

%% Compute BIC
disp('check nGroup')
sum(paramsBIC.nGroup)

X_train = X_train'; % Ntrain x k
[Ntrain, Ktrain]  = size(X_train)

% Compute Maximized value of the likelihood function
logdetcov = -0.5*logdet(paramsBIC.PooledCov); 
otherConst = -0.5*log(2*pi);
TotalConst = Ntrain*(logdetcov + otherConst);
PriorTerm = paramsBIC.nGroup * paramsBIC.PriorProb';
MainTerm = 0; 
for i = 1: numel(classifier.ClassLabel)
     Group = (Labels_train == classifier.ClassLabel(i));
     centeredX = bsxfun(@minus, X_train(Group,:), paramsBIC.GroupMean(i));
     mainterm = centeredX*paramsBIC.invCov;
     MainTerm = MainTerm -0.5*sum(sum(mainterm.*centeredX, 2));
end  

L = TotalConst + PriorTerm + MainTerm; % Likelihood function: L(Data|Model); the joint probability of data and label 

% Determine the number of free parameters
numMean		= numel(paramsBIC.nGroup)
numPrior	= numel(paramsBIC.nGroup) 
numCov 		= 1;
numClassifierCoeff = numel(classifier.W) % (Dtrain+1)* ClassLabel
disp('check whether same')
(Dtrain + 1)* numel(paramsBIC.ClassLabel)
%numProjectionMatrixCoeff

BIC = -2*L + k*(log(n) - log(2*pi)); 
disp('hi')
% normalDist = -0.5*logdet(paramsBIC.PooledCov) - centeredX'*paramsBIC.invCov*centeredX - 
% normalDist = -0.5*logdet(paramsBIC.PooledCov) - (X_train(paramsBIC.Group,:)-paramsBIC.GroupMean)'*paramsBIC.invCov*


% normalDist = -0.5*(log(det(cov_mat)) - (X-mean)'*inv(cov_mat)*(X-mean) - D*log(2*pi));

% max_L = sum(normalDist*prior)
BIC = 0; 




end


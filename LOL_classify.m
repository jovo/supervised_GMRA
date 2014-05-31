function [Yhat, Proj, P, boundary] = LOL_classify(sample,training,group,task)
% 
% 
% DENL = LOL
% DRNL = DRDA
% NRNL = RDA
% NENL = PDA
% NNNN = NaiveBayes
% NNNL = LDA
% NNNQ = QDA
%% 


if ~isfield(task, 'GMRAClassifier')
    [transformers, deciders] = parse_algs(task.types);
else % For supervised_GMRA application
    task.types = task.GMRAClassifier;
    task.ntrain = size(training,2);
    task.ks=unique(floor(logspace(0,log10(task.ntrain),task.ntrain)));
    task.Kmax = max(task.ks);
    
    algs = task.types;
    Tchar=[];
    for i=1:length(algs)
        Tchar=[Tchar; algs{i}(1:3)];
    end
    Tchar=unique(Tchar,'rows');
    [Ntransformers,~]=size(Tchar);
    transformers=cell(Ntransformers,1);
    for i=1:Ntransformers
        transformers{i}=Tchar(i,:);
    end
    
    deciders{1}{1}='linear';
end

[Proj, P] = LOL(training',group,transformers,task.Kmax);

% [transformers, deciders] = parse_algs(task.types);
% [Proj, P] = LOL(training',group,transformers,task.Kmax);

Yhat=cell(length(task.types),1);
boundary = cell(length(task.types),1); % variable added for classify_single_node_train for supervised_GMRA
k=0;
for i=1:length(transformers)
    Xtest=Proj{i}.V*sample';
    Xtrain=Proj{i}.V*training';
    for j=1:length(deciders{i})
        k=k+1;
        [Yhat{k}, boundary] = decide(Xtest,Xtrain,group,deciders{i}{j},task.ks); % 'boundary' added for classify_single_node_train for supervised_GMRA
    end
end
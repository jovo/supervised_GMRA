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

% disp('aa')
if ~isfield(task, 'LOL_alg')
    [transformers, deciders] = parse_algs(task.types);
else % For supervised_GMRA application
    task.types = task.LOL_alg;
    task.Kmax = max(task.ks);
    algs{1} = task.types;
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
% disp('bb')
% size(training')
% size(group)
% task.Kmax
% LOL Input: data: D by N, group: 1 by N
group = double(group); % As LOL uses isnan
[Proj, P] = LOL(training',group,transformers,task.Kmax);
% disp('cc')
Yhat=cell(length(task.types),1);
boundary = cell(length(task.types),1); % variable added for classify_single_node_train for supervised_GMRA
% disp('checking the size of the projection matrix and the matrix, sample')
% size(Proj{1}.V)
% size(sample')
k=0;
for i=1:length(transformers)
    if ~isempty(sample)
        Xtest=Proj{i}.V*sample';
    else
        Xtest = [];
    end
    Xtrain=Proj{i}.V*training';
    for j=1:length(deciders{i})
        k=k+1;
        [Yhat{k}, boundary] = decide(Xtest,Xtrain,group,deciders{i}{j},task.ks); % 'boundary' added for classify_single_node_train for supervised_GMRA
    end
end

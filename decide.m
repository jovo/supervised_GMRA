function [Yhat, boundary] = decide(sample,training,group,classifier,ks)

Nks=length(ks);
siz=size(sample);
ntest=siz(2);
Yhat=nan(Nks,ntest);
boundary=cell(Nks, 1);
for i=1:Nks
    try
        if any(strcmp(classifier,{'linear','quadratic','diagLinear','diagquadratic','mahalanobis'}))
            [Yhat(i,:), ~, ~, ~, coef]  = classify(sample(1:ks(i),:)',training(1:ks(i),:)',group,classifier);
            boundary{i} = [coef(1,2).const; coef(1,2).linear]'; % Added output variable boundary
        elseif strcmp(classifier,'NaiveBayes')
            nb = NaiveBayes.fit(training',group);
            Yhat = predict(nb,sample')';
        elseif strcmp(classifier,'svm')
            SVMStruct = svmtrain(training',group);
            Yhat = svmclassify(SVMStruct,sample');
        elseif strcmp(task.algs{i},'RF')
            B = TreeBagger(100,training,group);
            [~, scores] = predict(B,sample');
            Yhat=scores(:,1)<scores(:,2);
        elseif strcmp(task.algs{i},'kNN')
            %             d=bsxfun(@minus,Z.Xtrain,Z.Xtest).^2;
            %             [~,IX]=sort(d);
            %             for l=1:tasks1.Nks
            %                 Yhat(i)=sum(Z.Ytrain(IX(1:l,:)))>k/2;
            %                 loop{k}.out(i,l) = get_task_stats(Yhat,Z.Ytest);              % get accuracy
            %             end
        end
    catch err
        if i>1
            display(['the ', classifier, ' classifier barfed during embedding dimension ', num2str(ks(i))])
        else
            display(['the ', classifier, ' classifier barfed '])
        end
        display(err.message)
        break
    end
end
function [Yhat, boundary] = decide(sample,training,group,classifier,ks)
disp('start of decide')
Nks=length(ks);
siz=size(sample);
ntest=siz(2);
Yhat=nan(Nks,ntest);
boundary=cell(Nks, 1);
for i=1:Nks
     try
        if any(strcmp(classifier,{'linear','quadratic','diagLinear','diagquadratic','mahalanobis'}))
            if isempty(sample)
                sample_node = ones(1,ks(i));
            else
		disp('disp the size of the sample and ks')
		size(sample)
		ks(i)
                sample_node = sample(1:ks(i),:)';
            end
            if ~isempty(sample_node)
        %       [Yhat(i,:), ~, ~, ~, coef]  = classify(sample_node,training(1:ks(i),:)',group,classifier);
		disp('final input to the classifier: LDA_traintest')
		size(training(1:ks(i),:))
		size(group)
		size(sample_node')
		disp('end')
		[Yhat(i,:), ~, coef, ~] = LDA_traintest(training(1:ks(i),:),group,sample_node', [] );
	    else	
		disp('sample_node empty!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            end    
            if numel(unique(group)) > 1
        %        boundary{i} = [coef(1,2).const; coef(1,2).linear]'; % Added output variable boundary
         	boundary{i} = coef;
	    else 
                boundary{i} = Inf;
                disp('Alert!!!!!! Boundary set as inf because there was only one unique label in the node.')
            end
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
disp('end of decide.m')
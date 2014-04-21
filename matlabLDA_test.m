function [n_errors, labels_pred, labels_prob] = matlabLDA_test( classifier, data_test, labels_test )

if ~isempty(data_test) && ~isempty(classifier.W),    
    % Use the model on test set
    L = [ones(size(data_test,2),1) data_test'] * classifier.W';            
%     P = exp(L) ./ repmat(sum(exp(L),2),[1 size(L,2)]);        
%     [~,labels_pred] = max(L,[],2);  
    labels_pred = (L< 0) + 1;
%     labels_prob = max(P,[],2);  
    labels_prob = [];
    labels_pred
    classifier.ClassLabel
    labels_pred = classifier.ClassLabel(labels_pred);
    if (nargin>2) && (~isempty(labels_test)),
        n_errors = sum(labels_pred ~= labels_test');
    else
        n_errors = NaN;
    end;
else
    if isempty(classifier.W),
            dbstop,          %DBG
    end;
    labels_prob = [];
    labels_pred = [];
    n_errors    = 0;
end;

return;

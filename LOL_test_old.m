function [n_errors, labels_pred, labels_prob] = LOL_test( classifier, data_test, labels_test )

% 
% disp('hello') 
% size(labels_pred{1})
% size(labels_test)
% for i = 1: size(labels_pred{1},1)
% ne(i) = sum(labels_pred{1}(i,:) ~= labels_test);
% end
% disp('world')
% [n_errors, mse_ks] = min(ne);
% labels_pred = labels_pred{1}(mse_ks, :);
% labels_prob = 0;

% labels_pred_ks = nan(size(classifier.W,1), size(data_test{1},2)); % #ks by metis-partitioned-n 
% n_errors_ks = nan(size(classifier.W,1), 1); % #ks by 1
% disp('should not be cell')
% size(data_test)
if ~isempty(data_test) && ~isempty(classifier.W),
    % Use the model on test set
    for i = 1: size(classifier.W,1)
        %         data_test_k = data_test{i}; % 1 by N_lol = 40
        %         classifier_k = classifier.W{i}; % 1 by 2
%         size(data_test)
%	size([ones(size(data_test,2),1) data_test'])
%         size(classifier.W{1}')
%         i
	
        L = [ones(size(data_test,2),1) data_test'] * classifier.W';
        % metis-partitioned-n by lol-reduced-p +1  *  lol-reduced-p +1 by 1
        P = exp(L) ./ repmat(sum(exp(L),2),[1 size(L,2)]);
        labels_prob = max(P,[],2);
        labels_pred = (L< 0) + 1;
        labels_pred = classifier.ClassLabel(labels_pred);
        if (nargin>2) && (~isempty(labels_test)),
            	disp('checking the size of labels_pred and labels_test_transpose')
		size(labels_pred)
		size(labels_test')
		n_errors = sum(labels_pred ~= labels_test');
        else
            n_errors = NaN;
        end;
        %         labels_pred = classifier.ClassLabel(((L< 0) + 1))';
        %         n_errors = sum(labels_pred_ks(i,:) ~= labels_test);
        %         [n_errors, minerr_ks] = min(n_errors_ks);
        %         labels_pred = labels_pred_ks(minerr_ks, :);
        %         if ~((nargin>2) && (~isempty(labels_test))),
        %             n_errors = NaN;
        %         end;
    end
else
    if isempty(classifier.W),
        dbstop,          %DBG
    end;
    labels_prob = [];
    labels_pred = [];
    n_errors    = 0;
end;
return;

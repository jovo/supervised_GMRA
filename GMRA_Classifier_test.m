function ClassifierResults = GMRA_Classifier_test( MRA, X, TrainGroup, Y, classifier, Opts )

%
% function MRA = GMRA_Classifier_test( MRA, X, TrainGroup, Y )
%
% IN:
%   MRA         : a GMRA, with GMRA_Classifier called on some training set (typically with the same X,TrainGroup,Y as this function)
%   X           : D by N data
%   TrainGroup  : N row vector, 0 indiciating test, 1 indicating training
%   Y           : N row vector of labels
%
% OUT:
%   MRA with the following additional fields:
%       Data_test_GWT   : the GWT of the test data
%       Classifier.Test : structure with the following fields:
%                           errors              : vector of length #(active nodes in the classifier) with the prediction error in each active node
%                           Labels_node_pred    : cell array of length #(active nodes in the classifier) with predicted labels in each active node
%                           Labels              : row vector of length #test labels with the predicted labels
%
% (c) Copyright Mauro Maggioni, 2013

Timing.GMRAClassifierTest = cputime;

global COMBINED
% disp('Starting the GMRA_Classifier_test....')
% disp('checking the MRA.min_ks input for GMRA_Classifier_test.................................');
% MRA.min_ks

if isempty(MRA.Classifier.activenode_idxs)
    ClassifierResults = [];
    disp('ERROR: No active node in the trained GMRA-classifier.');
    
else
    if nargin < 5
        classifier = @LDA_test;
    end
    
    fcn_test_single_node = @classify_single_node_test;
    Data_test           = X(:,TrainGroup==0);
    MRA.Data_test_GWT   = FGWT( MRA , Data_test );                      % The classifier works on the GWT side
    MRA.Data_test       = Data_test;                                    % Input of Data_test to test classifier on the data side
    if isfield(MRA, 'X_test')
        MRA.Data_test = MRA.X_test;
    end
    MRA.Labels_test     = Y(TrainGroup==0);
    %   disp('length of the activnodes')
    %  length(MRA.Classifier.activenode_idxs)
    % Go through the active nodes in the classifier and classify the test points in there
    for k = 1:length(MRA.Classifier.activenode_idxs),
        current_node_idx = MRA.Classifier.activenode_idxs(k);
        %       min_ks = MRA.min_ks(current_node_idx)
        % Classify on the k-th active node, using its corresponding classifier
        [ClassifierResults.Test.errors(current_node_idx),ClassifierResults.Test.Labels_node_pred{current_node_idx},dataIdxs_test, ...
            ClassifierResults.Test.Labels_node_prob{current_node_idx}] = ...
            fcn_test_single_node( MRA, MRA.Data_test_GWT, MRA.Labels_test, struct('classifier', classifier, 'current_node_idx',current_node_idx, ...
            'LOL_alg',Opts.LOL_alg, 'COMBINED',COMBINED, 'UseX', Opts.UseX) );
        
        if isequal(classifier, @LOL_test)
            
            % String the predicted labels in an easily accessible vector
            ClassifierResults.Test.Labels.labels_pred{k}    = ClassifierResults.Test.Labels_node_pred{current_node_idx};
            ClassifierResults.Test.Labels.idx{k} 		= dataIdxs_test;
            ClassifierResults.Test.LabelsProb(dataIdxs_test) = NaN;
            temp = Y(TrainGroup == 0);
            temp = temp(dataIdxs_test);
            % for
            %	disp('dispaying temp1 and temp2')
            %	temp
            %	size(temp)
            temp2 = ClassifierResults.Test.Labels.labels_pred{k};
            %	temp2(1:10)
            %	for i = 1: numel(temp2)
            %		temp2{i}
            %	end
            %	size(temp2)
            if ~isempty(temp)
                for i = 1: numel(temp2)
                    % temp2{i}
                    %		size(temp2{i})
                    %		size(temp)
                    ClassifierResults.Test.Labels.errors(k,i) = sum(temp2{i} ~= temp');
                end
            end
        else
            % String the predicted labels in an easily accessible vector
            ClassifierResults.Test.Labels(dataIdxs_test)     = ClassifierResults.Test.Labels_node_pred{current_node_idx};
            ClassifierResults.Test.LabelsProb(dataIdxs_test) = ClassifierResults.Test.Labels_node_prob{current_node_idx};
        end
    end;
    % Y(TrainGroup==0)
    % ClassifierResults.Test.Labels.labels_pred
end

ClassifierResults.Timing.GMRAClassifierTest = cputime-Timing.GMRAClassifierTest;
return;

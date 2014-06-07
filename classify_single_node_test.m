function [total_errors, labels_pred, dataIdxs_test, labels_prob] = classify_single_node_test( MRAClassifier, Data_test_GWT, Labels_test, Opts )

%
% IN:
%   Data_test_GWT   : GWT of test data
%   Labels_train    : labels of training data. Row vector.
%   Opts:
%       Opts            : structure contaning the following fields:
%                           [classifier]     : function handle to classifier. Default: LDA.
%                           Opts.current_node_idx : index at which to train the classifier
%                           [Opts.COMBINED]       : whether to use only scaling function subspace (=0) or also wavelet subspace (=1). Default: 0.
%
% OUT:
%   trained_classifier : classifier
%

if ~isfield(Opts,'classifier') || isempty(Opts.classifier),     Opts.classifier     = @QDA_test;    end;
if ~isfield(Opts,'COMBINED')   || isempty(Opts.COMBINED),       Opts.Opts.COMBINED  = 0;            end;
disp('The classifier for the test data: ')
Opts.classifier

% Test data
if ~Opts.COMBINED %|| Opts.current_node_idx == length(MRA.cp),
    coeffs_test = cat(1, Data_test_GWT.CelScalCoeffs{Data_test_GWT.Cel_cpidx == Opts.current_node_idx})';
else
    coeffs_test = cat(2, cat(1, Data_test_GWT.CelScalCoeffs{Data_test_GWT.Cel_cpidx == Opts.current_node_idx}), cat(1,Data_test_GWT.CelWavCoeffs{Data_test_GWT.Cel_cpidx == Opts.current_node_idx}))';
end
dataIdxs_test      = Data_test_GWT.PointsInNet{Opts.current_node_idx};
dataLabels_test    = Labels_test(dataIdxs_test);

% Test 
if ~isempty(MRAClassifier.Classifier.Classifier{Opts.current_node_idx}),
    if isequal(Opts.classifier, @LOL_test)
        task = {};
        task.LOL_alg = Opts.LOL_alg;
        %         task.ntrain = cp.TrainSize(1);
        size(coeffs_test)
        task.ntrain = size(coeffs_test, 1);
        ks=unique(floor(logspace(0,log10(task.ntrain),task.ntrain)))
        Opts.task = task;
        for i = 1:length(ks)
            Opts.task.ks = ks(i);
            %         [~,~,classifier_trained{i}] = Opts.Classifier( coeffs_train', dataLabels_train,[],[], Opts );
            classifier_input = MRAClassifier.Classifier.Classifier{Opts.current_node_idx};
%             whos classifier_input
%             size(classifier_input)
%             length(ks)
%             classifier_input{1}
%             classifier_input{2}
%             disp('checking the dimension for projection of test data')
%             size(coeffs_test)
%             size(classifier_input{i}.Proj{1}.V)
%             classifier_input{i}.Proj{1}
            coeffs_test_projd = classifier_input{i}.Proj{1}.V * coeffs_test;
%             size(coeffs_test_projd)
%             disp('checking the dimension for projection of test data')
            [total_errors, labels_pred, labels_prob] = Opts.classifier( classifier_input{i} , coeffs_test_projd, dataLabels_test );
%             total_errors
        end
%         total_errors = total_errors_ks{1};
    else
        [total_errors, labels_pred, labels_prob] = Opts.classifier( MRAClassifier.Classifier.Classifier{Opts.current_node_idx}, coeffs_test, dataLabels_test );
    end
    
elseif ~isempty(dataIdxs_test)
    labels_prob  = NaN(size(dataLabels_test))';
    labels_pred  = NaN(size(dataLabels_test))';
    total_errors = sum(dataLabels_test~=labels_pred');
else
    labels_prob  = [];
    labels_pred  = [];
    total_errors = 0;
end;

return;
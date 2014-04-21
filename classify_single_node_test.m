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
    [total_errors, labels_pred, labels_prob] = Opts.classifier( MRAClassifier.Classifier.Classifier{Opts.current_node_idx}, coeffs_test, dataLabels_test );
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
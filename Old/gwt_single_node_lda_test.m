function [total_errors, labels_pred, dataIdxs_test] = gwt_single_node_lda_test( MRA, Data_test_GWT, Labels_test, current_node_idx, COMBINED )

% COMBINED is boolean for whether to use Scal/Wav combined or not as basis for LDA

% Test data
if ~COMBINED || current_node_idx == length(MRA.cp),
    coeffs_test = cat(1, Data_test_GWT.CelScalCoeffs{Data_test_GWT.Cel_cpidx == current_node_idx})';
else
    coeffs_test = cat(2, cat(1, Data_test_GWT.CelScalCoeffs{Data_test_GWT.Cel_cpidx == current_node_idx}), cat(1,Data_test_GWT.CelWavCoeffs{Data_test_GWT.Cel_cpidx == current_node_idx}))';
end
dataIdxs_test      = Data_test_GWT.PointsInNet{current_node_idx};
dataLabels_test    = Labels_test(dataIdxs_test);

% Test 
if ~isempty(MRA.Classifier.Model{current_node_idx}),
    [total_errors, labels_pred] = lda_test( MRA.Classifier.Model{current_node_idx}, MRA.Classifier.ModelLabels{current_node_idx}, coeffs_test, dataLabels_test );
elseif length(dataIdxs_test)>0
    labels_pred  = NaN(size(dataLabels_test))';%MRA.Classifier.ModelLabels{current_node_idx}*ones(size(dataLabels_test))';
    total_errors = sum(dataLabels_test~=labels_pred');
else
    labels_pred  = [];
    total_errors = 0;
end;

end
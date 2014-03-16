function [model_train,model_labels,dataIdxs_train] = classify_single_node( Data_train_GWT, Labels_train, current_node_idx, COMBINED )

% COMBINED is boolean for whether to use Scal/Wav combined or not as basis
%   for LDA
%

if ~COMBINED %|| current_node_idx == length(GWT.cp),
    coeffs_train = cat(1, Data_train_GWT.CelScalCoeffs{Data_train_GWT.Cel_cpidx == current_node_idx})';
else
    coeffs_train = cat(2, cat(1, Data_train_GWT.CelScalCoeffs{Data_train_GWT.Cel_cpidx == current_node_idx}), cat(1,Data_train_GWT.CelWavCoeffs{Data_train_GWT.Cel_cpidx == current_node_idx}))';
end
dataIdxs_train      = Data_train_GWT.PointsInNet{current_node_idx};
dataLabels_train    = Labels_train(dataIdxs_train);

if length(dataLabels_train)>1,
    [model_train,model_labels] = LDA( coeffs_train', dataLabels_train' );
else
    model_train     = [];
    model_labels    = NaN;
end;

return;
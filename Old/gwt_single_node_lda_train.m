function [model_train,model_labels,dataIdxs_train] = gwt_single_node_lda_train( GWT, Data_train, Labels_train, current_node_idx, COMBINED )
% Uses Data for Cel_cpidx, CelScalCoeffs, CelWavCoeffs
% COMBINED is boolean for whether to use Scal/Wav combined or not as basis
%   for LDA
%

if ~COMBINED || current_node_idx == length(GWT.cp),
    coeffs_train = cat(1, Data_train.CelScalCoeffs{Data_train.Cel_cpidx == current_node_idx})';
else
    coeffs_train = cat(2, cat(1, Data_train.CelScalCoeffs{Data_train.Cel_cpidx == current_node_idx}), cat(1,Data_train.CelWavCoeffs{Data_train.Cel_cpidx == current_node_idx}))';
end
dataIdxs_train      = Data_train.PointsInNet{current_node_idx};
dataLabels_train    = Labels_train(dataIdxs_train);

if length(dataLabels_train)>1,
    [model_train,model_labels] = LDA( coeffs_train', dataLabels_train' );
else
    model_train     = [];
    model_labels    = NaN;
end;

return;
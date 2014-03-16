function [complexity] = gwt_single_node_complexity( GWT, Data, imgOpts, current_node_idx, COMBINED )
% Uses GWT for cp and PointsInNet
% Uses Data for Cel_cpidx, CelScalCoeffs, CelWavCoeffs
% COMBINED is boolean for whether to use Scal/Wav combined or not as basis
%   for LDA
% 
% NOTE: The arguments list for this have become non-parallel with
% gwt_single_node_lda_crossvalidation...

    % Train data
    if ~COMBINED || current_node_idx == length(GWT.cp),
        coeffs = cat(1, Data.CelScalCoeffs{Data.Cel_cpidx == current_node_idx})';
    else
        coeffs = cat(2, cat(1, Data.CelScalCoeffs{Data.Cel_cpidx == current_node_idx}), cat(1,Data.CelWavCoeffs{Data.Cel_cpidx == current_node_idx}))';
    end
    dataIdxs = GWT.PointsInNet{current_node_idx};
    dataLabels = imgOpts.Labels_train(dataIdxs);

    dimensionality = size(coeffs,1);
    n_cats = length(unique(dataLabels));
    
    complexity = n_cats * dimensionality^2;
    
end
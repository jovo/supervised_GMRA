function [total_errors, std_errors] = gwt_single_node_lda_crossvalidation( GWT, Data, Labels_train, current_node_idx, COMBINED )
% Uses GWT for cp and PointsInNet
% Uses Data for Cel_cpidx, CelScalCoeffs, CelWavCoeffs
% COMBINED is boolean for whether to use Scal/Wav combined or not as basis for LDA

if ~COMBINED || current_node_idx == length(GWT.cp),
    coeffs = cat(1, Data.CelScalCoeffs{Data.Cel_cpidx == current_node_idx})';
else
    coeffs = cat(2, cat(1, Data.CelScalCoeffs{Data.Cel_cpidx == current_node_idx}), cat(1,Data.CelWavCoeffs{Data.Cel_cpidx == current_node_idx}))';
end
dataLabels = Labels_train(Data.PointsInNet{current_node_idx});          %dataLabels = Labels_train(GWT.PointsInNet{current_node_idx});              % MM changed: imgOpts.Labels(dataIdxs);

node_pts  = length(dataLabels);
node_cats = length(unique(dataLabels));


if (node_cats>1) && (node_pts>1) && size(Data,1)>0
    cp = cvpartition(dataLabels,'k',10);
    classf = @(xtrain, ytrain,xtest)(LDA_traintest(xtrain',ytrain',xtest',[]));
    cvMCR = crossval('mcr',coeffs',dataLabels','predfun', classf,'partition',cp); 
    total_errors    = cvMCR*length(dataLabels);
    std_errors      = 0;
else
    total_errors = Inf;
    std_errors   = Inf;
end;

return;
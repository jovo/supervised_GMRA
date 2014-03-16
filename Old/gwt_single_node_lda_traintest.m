function [total_errors, model_train,labels_pred,labels_train] = gwt_single_node_lda_traintest( GWT, Data_train, Data_test, imgOpts, current_node_idx, COMBINED )
% Uses GWT for cp and PointsInNet
% Uses Data for Cel_cpidx, CelScalCoeffs, CelWavCoeffs
% COMBINED is boolean for whether to use Scal/Wav combined or not as basis
%   for LDA
%
% NOTE: The arguments list for this have become non-parallel with
% gwt_single_node_lda_crossvalidation...

% Train data
if ~COMBINED || current_node_idx == length(GWT.cp),
    coeffs_train = cat(1, Data_train.CelScalCoeffs{Data_train.Cel_cpidx == current_node_idx})';
else
    coeffs_train = cat(2, cat(1, Data_train.CelScalCoeffs{Data_train.Cel_cpidx == current_node_idx}), cat(1,Data_train.CelWavCoeffs{Data_train.Cel_cpidx == current_node_idx}))';
end
%dataIdxs_train = GWT.PointsInNet{current_node_idx};
dataIdxs_train = Data_train.PointsInNet{current_node_idx};
dataLabels_train = imgOpts.Labels_train(dataIdxs_train);

node_pts_train = length(dataLabels_train);
node_cats_train = length(unique(dataLabels_train));

% Test data
if ~COMBINED || current_node_idx == length(GWT.cp),
    coeffs_test = cat(1, Data_test.CelScalCoeffs{Data_test.Cel_cpidx == current_node_idx})';
else
    coeffs_test = cat(2, cat(1, Data_test.CelScalCoeffs{Data_test.Cel_cpidx == current_node_idx}), cat(1,Data_test.CelWavCoeffs{Data_test.Cel_cpidx == current_node_idx}))';
end
% NOTE: Non-parallel input data structures for train and test
dataIdxs_test = Data_test.PointsInNet{current_node_idx};
dataLabels_test = imgOpts.Labels_test(dataIdxs_test);

node_pts_test = length(dataLabels_test);
node_cats_test = length(unique(dataLabels_test));

% Run
% Must have more than one training point and category to make a
% reasonable model.
if (node_cats_train > 1 && node_pts_train > 1 && node_cats_test >= 1 && node_pts_test >= 1),
    [total_errors, model_train, labels_pred] = lda_traintest( coeffs_train, dataLabels_train, coeffs_test, dataLabels_test );
elseif (node_cats_train==1) || (node_pts_test==0),    
    model_train = [];
    if node_cats_train==1 && node_pts_test>0,
        labels_pred = unique(dataLabels_train)*ones(length(dataLabels_test),1);
        total_errors = sum(labels_pred~=dataLabels_test');
    else
        labels_pred = [];
        total_errors = 0;
    end;
else
    total_errors = node_pts_test;
    model_train = [];
    labels_pred = [];
    fprintf('\n NOT SURE THIS SHOULD HAPPEN!');
end;

%     elseif (node_cats_train > 1 && node_pts_train > 1 && node_cats_test == 0 && node_pts_test == 0),
%         total_errors = 0;
%         model_train = lda_modelonly( coeffs_train, dataLabels_train );
%     else
%         total_errors = inf;
%         % NOTE: Not sure this is the right choice for "no model"...
%         model_train = 0;
%     end

end
function [classifier_trained,dataIdxs_train] = classify_single_node_train( Data_train_GWT, Labels_train, Opts )
%
% IN:
%   Data_train_GWT  : GWT of training data
%   Labels_train    : labels of training data. Row vector.
%   Opts:
%       Opts            : structure contaning the following fields:
%                           [Classifier]     : function handle to classifier. Default: LDA_traintest.
%                           current_node_idx : index at which to train the classifier
%                           [COMBINED]       : whether to use only scaling function subspace (=0) or also wavelet subspace (=1). Default: 0.
%
% OUT:
%   trained_classifier : classifier
%

if ~isfield(Opts,'Classifier') || isempty(Opts.Classifier),     Opts.Classifier = @LDA_traintest;   end;
if ~isfield(Opts,'COMBINED')   || isempty(Opts.COMBINED),       Opts.COMBINED   = 0;                end;

if ~Opts.COMBINED %|| current_node_idx == length(Data_train.PointsInNet),
    coeffs_train = cat(1, Data_train_GWT.CelScalCoeffs{Data_train_GWT.Cel_cpidx == Opts.current_node_idx});
else
    coeffs_train = cat(2,   cat(1, Data_train_GWT.CelScalCoeffs{Data_train_GWT.Cel_cpidx == Opts.current_node_idx}), ...
                            cat(1, Data_train_GWT.CelWavCoeffs{Data_train_GWT.Cel_cpidx == Opts.current_node_idx}));
end
dataIdxs_train      = Data_train_GWT.PointsInNet{Opts.current_node_idx};
dataLabels_train    = Labels_train(dataIdxs_train);

[~,~,classifier_trained] = Opts.Classifier( coeffs_train', dataLabels_train,[],[], Opts );

return;
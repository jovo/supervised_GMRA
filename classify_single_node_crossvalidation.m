function [total_errors, std_errors, min_ks] = classify_single_node_crossvalidation( Data_GWT, Labels_train, Opts )

%
% IN:
%   Data_train_GWT  : GWT of training data
%   Labels_train    : labels of training data. Row vector.
%   Opts:
%       Opts            : structure contaning the following fields:
%                           [Classifier]     : function handle to classifier. Default: LDA_train.
%                           current_node_idx : index at which to train the classifier
%                           [COMBINED]       : whether to use only scaling function subspace (=0) or also wavelet subspace (=1). Default: 0.
%
% OUT:
%   trained_classifier : classifier
%

if ~isfield(Opts,'Classifier') || isempty(Opts.Classifier),     Opts.Classifier = @LDA_traintest;   end;
if ~isfield(Opts,'COMBINED')   || isempty(Opts.COMBINED),       Opts.COMBINED   = 0;                end;

if ~Opts.COMBINED %|| current_node_idx == length(GWT.cp),
    coeffs = cat(1, Data_GWT.CelScalCoeffs{Data_GWT.Cel_cpidx == Opts.current_node_idx})';
else
    coeffs = cat(2, cat(1, Data_GWT.CelScalCoeffs{Data_GWT.Cel_cpidx == Opts.current_node_idx}), cat(1,Data_GWT.CelWavCoeffs{Data_GWT.Cel_cpidx == Opts.current_node_idx}))';
end
dataLabels = Labels_train(Data_GWT.PointsInNet{Opts.current_node_idx});

node_pts  = length(dataLabels);
node_cats = length(unique(dataLabels));

if (node_cats>1) && (node_pts>node_cats) && size(Data_GWT,1)>0
    % Perform crossvalidation
    cp = cvpartition(dataLabels,'k',10);
    opts = statset('UseParallel','never');                                                     % Matlab parallel CV is buggy. What a piece of junk.
    if isequal(Opts.Classifier, @LOL_traintest)
	[task, ks] = set_task_LOL(Opts, size(coeffs,1));
	Opts.task = task;
	
	N = size(coeffs, 2);

	% As here we use one-time check instead of crossval within training data for computation time,
	% the ratio of the training vs. the test within training data here could affect the search for
	% the appropriate choice of nodes and the k.
	ratio = 7/10;
	ntest = floor(ratio*N);
	ntrain = N-ntest;
cv = 1;
CV_ERR_LOL = zeros(1, length(ks));
        for j = 1: cv
	% Swap again in case the data was not mixed well.
	swp_idx = randperm(N);
	coeffs_swp = coeffs(:,swp_idx);
	dataLabels_swp = dataLabels(swp_idx);
	
	data_test = coeffs_swp(:, 1: ntest);
	data_train = coeffs_swp(:, ntest+1: end);
	labels_test = dataLabels_swp(1: ntest);
	labels_train = dataLabels_swp(ntest+1: end);
        
	Opts.task.ks = ks;	
	[labels_pred_LOL, n_errors_LOL, classifier_LOL, ~] = LOL_traintest( data_train , labels_train, data_test, labels_test, Opts );
 	
            for i = 1:length(ks)
%            Opts.task.ks = ks(i);
             ERR_LOL(i) = sum(labels_pred_LOL(:,i) ~= labels_test');
%		n_errors_LOL{i}
%		ERR_LOL(i)
%            classf = @(xtrain, ytrain,xtest)(Opts.Classifier(xtrain',ytrain',xtest',[],Opts));
%            cvMCR(i) = crossval('mcr',coeffs_swp',dataLabels_swp','predfun', classf,'partition',cp,'Options',opts)
%   	    total_errors_ks(i)    = cvMCR(i)*length(dataLabels);
            end

        CV_ERR_LOL = CV_ERR_LOL + ERR_LOL;
        end
       [total_errors, min_ks] = min(CV_ERR_LOL);
	min_ks = ks(min_ks);

    else
        classf = @(xtrain, ytrain,xtest)(Opts.Classifier(xtrain',ytrain',xtest',[],Opts));
        cvMCR = crossval('mcr',coeffs',dataLabels','predfun', classf,'partition',cp,'Options',opts)
        total_errors   = cvMCR*length(dataLabels);
	min_ks = 0;
    end
    std_errors      = 0;
else
    total_errors = Inf;
    std_errors   = Inf;
    min_ks = Inf;
end;
disp('total_errors')
total_errors
return;

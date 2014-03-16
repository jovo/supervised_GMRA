function [model] = lda_modelonly( data_train, labels_train, varargin )

% Cross-validation (1/5 holdout for now) of Linear Discriminant Analysis
%
% data = [d n] array of measurements
% labels = [1 n] array of category labels (integers)
%
% NOTE: If there are no test points, model is returned and n_errors == 0
%
% Requires Will Dwinnell's LDA code
%   http://matlabdatamining.blogspot.com/2010/12/linear-discriminant-analysis-lda.html
%   http://www.mathworks.com/matlabcentral/fileexchange/29673-lda-linear-discriminant-analysis

%% Utility function

    function [tf] = allintarray(xx)
        tf = false;
        % numeric
        if ~isnumeric(xx),
            return;
        end
        % 1d array
        if size(xx,1) ~= 1,
            return;
        end
        % integers
        for idx = 1:length(xx),
            if uint8(xx(idx)) ~= xx(idx),
                return;
            end
        end
        tf = true;
    end

%% Argument parsing

% Need value to be an integer, but isinteger() tests for int array...
checkDigitsArray = @(x) allintarray(x);
    
p = inputParser;

% NOTE: These have to be added in the arglist order!!
addRequired(p, 'data_train', @isnumeric);
addRequired(p, 'labels_train', checkDigitsArray);

parse(p, data_train, labels_train, varargin{:});

meas_train = p.Results.data_train;
cats_train = p.Results.labels_train;

%% do lda on train and then test on same model

% LDA code wants measurements [n d] order
model = LDA(meas_train', cats_train');


end

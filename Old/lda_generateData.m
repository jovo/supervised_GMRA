function [meas, labels] = lda_generateData( data_name, varargin )

% INPUT:
%
% data_name required string describing data set
% validDataNames = {'iris', 'scinews', 'MNIST_Digits'};

% optional:
% dim : dimensionality to reduce to with svd
%  (for MNIST only)
% digits : array of integers 0-9 for which digits to pick
% n_ea_digit : number of each digit to pick
%
% OUTPUT:
%
% meas = [d n] array of measurements
% labels = [1 n] array of category labels (integers)

% e.g. [meas, labels] = lda_generateData( 'MNIST_Digits', 'dim', 50, 'digits', [3 4 5], 'n_ea_digit', 250);

%% Utility function

    function [tf] = allintarray(xx)
        tf = false;
        if ~isnumeric(xx),
            return;
        end
        if size(xx,1) ~= 1,
            return;
        end
        for idx = 1:length(xx),
            if uint64(xx(idx)) ~= xx(idx),
                return;
            end
        end
        tf = true;
    end

%% Argument parsing

% Parameter value defaults
% dim = 0 is a flag for not doing any svd dimensionality reduction
dim = 0;
digits = [1 2];
n_ea_digit = 1000;

validDataNames = {'iris', 'sciencenews', 'mnist_digits'};
checkDataName = @(x) any(validatestring(lower(x), validDataNames));
    
checkDigitsArray = @(x) allintarray(x);
p = inputParser;

addRequired(p, 'data_name', checkDataName);
addParamValue(p, 'dim', dim, @isnumeric);
addParamValue(p, 'digits', digits, checkDigitsArray);
addParamValue(p, 'n_ea_digit', n_ea_digit, checkDigitsArray);

parse(p, data_name, varargin{:});

%% Generate Data

switch(lower(p.Results.data_name))
    
    case 'iris'
        
        S = load('fisheriris');

        % This data comes in as 
        % meas = [150 4] double measurements
        % species = {150 1} cell array of strings
        
        meas = S.meas';

        % Change species strings to integer category labels
        spec = unique(S.species); 
        labels = zeros(1,length(S.species)); 
        for ii = 1:length(S.species), 
            labels(ii) = find(ismember(spec,S.species(ii)) == 1); 
        end

    case 'sciencenews'
        
        S = load('X20');
        
        labels = S.classes(:,1);
        labels = labels(labels > 0)';
        X0 = S.X(labels > 0, :)';
        
        % Reduce dimensionality with randomized PCA
        if (p.Results.dim > 0),
            cm = mean(X0,2);
            X = X0 - repmat(cm, 1, size(X0,2));
            [~,S,V] = randPCA(X, p.Results.dim);
            X = S*V';
            meas = X;
        else
            meas = X0;
        end;
        
  case 'mnist_digits'

        [X0, ~, ~, labels] = GenerateDataSets( 'BMark_MNIST', ...
            struct('NumberOfPoints', p.Results.n_ea_digit,'AutotuneScales', false,'MnistOpts', struct('Sampling', 'FirstN', 'QueryDigits', p.Results.digits, 'ReturnForm', 'vector')));
        
        % try randomizing order...
        idxs = randperm(size(X0,2));
        labels = labels(idxs)';
        X0 = X0(:,idxs);

        % Reduce dimensionality with randomized PCA
        if (p.Results.dim > 0),
            cm = mean(X0,2);
            X = X0 - repmat(cm, 1, size(X0,2));
            [~,S,V] = randPCA(X, p.Results.dim);
            X = S*V';
            meas = X;
        else
            meas = X0;
        end;

end

end
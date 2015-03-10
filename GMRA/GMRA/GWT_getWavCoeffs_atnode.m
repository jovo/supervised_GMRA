function [WavCoeffs,Wav_cp_idx,ScalCoeffs] = GWT_getWavCoeffs_atnode( gMRA, DataGWT, cp_idx )

%
% function [WavCoeffs,Wav_cp_idx,ScalCoeffs] = GWT_getWavCoeffs_atnode( gMRA, DataGWT, cp_idx )
%
% IN:
%   gMRA        : GMRA structure as constructed by GMRA
%   DataGWT     : GWT transform of the data used to construct gMRA, as returned by FGWT_TrainingData
%   cp_idx      : cp index of a node in gMRA associated with a scaling subspace, can be a LeafNode
%
% OUT:
%   WavCoeffs   : cell array of wavelet coefficients, one cell for each of the wavelet subspaces child
%                 of the scaling function space requested
%   Wav_cp_idx  : cp indices of the wavelet subspaces in WavCoeffs
%   ScalCoeffs  : scaling functions coefficients in the cp_idx scaling subspace
%

% (c) Duke University
% Mauro Maggioni and Guangliang Chen
%

% Initialization
ScalCoeffs_k_idxs = [];

% Find the wavelet subspaces which are children of the given scaling space
Wav_cp_idx = find(gMRA.cp==cp_idx);
if isempty(Wav_cp_idx),                                                                         % It's a leafnode
    WavCoeffs   = {};
    Wav_cp_idx  = [];
    Scal_cp_idx = cp_idx;
    [k_idxs,j_idxs] = find(DataGWT.Cel_cpidx==cp_idx);
    ScalCoeffs{1} = DataGWT.CelScalCoeffs{k_idxs,j_idxs};
else
    Scal_cp_idx = [];
    WavCoeffs  = cell(1,length(Wav_cp_idx));
    if nargout>2, ScalCoeffs = cell(1,length(Wav_cp_idx)); end;
    
    % Go through the wavelet subspaces and collect the wavelet coefficients together with the corresponding scaling coefficients
    for k = 1:length(Wav_cp_idx),
        [k_idxs,j_idxs] = find(DataGWT.Cel_cpidx==Wav_cp_idx(k));
        WavCoeffs{k} = [];
        % The wavelet subspace appears in multiple cells in DataGWT: loop through those and collect the wavelet coefficients
        for i = 1:length(k_idxs),
            lNewWavCoeffs = DataGWT.CelWavCoeffs{k_idxs(i),j_idxs(i)};
            WavCoeffs{k}  = [WavCoeffs{k};lNewWavCoeffs];
            if nargout>2,
                ScalCoeffs{k}= [ScalCoeffs{k};DataGWT.CelScalCoeffs{k_idxs(i),j_idxs(i)-1}];                
            end;
        end;
    end;
end;

return;
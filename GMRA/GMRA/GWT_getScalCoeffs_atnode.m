function [ScalCoeffs,tangentialCorrections,k_idxs,j_idxs] = GWT_getScalCoeffs_atnode( gMRA, DataGWT, cp_idx )

% Find the scaling coefficients of points belonging to each node at scale j
[k_idxs,j_idxs] = find(DataGWT.Cel_cpidx==cp_idx);
ScalCoeffs = [];
tangentialCorrections = [];

for i = 1:length(k_idxs),
    ScalCoeffs            = [ScalCoeffs,DataGWT.CelScalCoeffs{k_idxs(i),j_idxs(i)}'];
    if isfield(DataGWT,'CelTangCoeffs'),
        tangentialCorrections = [tangentialCorrections,DataGWT.CelTangCoeffs{k_idxs(i),j_idxs(i)}'];
    end;
end;

return;
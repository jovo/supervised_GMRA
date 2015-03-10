function [nonzeroCoeffs, allCoeffs] = computeCoefficientCost(Data, gW, idxs)

nonzeroCoeffs = 0;
allCoeffs = 0;
    
if nargin<2, % all nodes are assumed
    
    for k = 1:size(Data.CelWavCoeffs,1)  
        C = [Data.CelWavCoeffs{k,:}]; % one leaf node, all scales
        nonzeroCoeffs = nonzeroCoeffs+sum(sum(abs(C)>0));
        allCoeffs = allCoeffs + numel(C);
    end
    
else %==3, for a subset of the nodes
    
    for k = 1:length(idxs)
        C = get_wavCoeffs_node(gW, Data, idxs(k)); % one node, and corresponding scale
        nonzeroCoeffs = nonzeroCoeffs+sum(sum(abs(C)>0));
        allCoeffs = allCoeffs + numel(C);
    end
    
end


return;
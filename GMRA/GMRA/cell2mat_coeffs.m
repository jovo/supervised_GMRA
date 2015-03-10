function [MatWavCoeffs maxWavDims MatWavDims] = cell2mat_coeffs(CelWavCoeffs, cumSzBlocks)

J = size(CelWavCoeffs,2);

if nargin<2
    szBlocks = cellfun(@(B)size(B,1), CelWavCoeffs(:,1));
    cumSzBlocks = cumsum(szBlocks);
end

N = cumSzBlocks(end);

MatWavCoeffs = cell(1,J); 
maxWavDims = zeros(1,J); % maximum wavelet dimension at each scale
MatWavDims = zeros(N,J); % element by scale 

for j = 1:J
    
    BlockWavDims_j = cellfun(@(B)size(B,2), CelWavCoeffs(:,j)); % each block is a leaf node
    maxWavDims(j) = max(BlockWavDims_j);
    MatWavCoeffs{j} = zeros(N,maxWavDims(j));
    
    positiveWavDims = (find(BlockWavDims_j>0))';
    
    if ~isempty(positiveWavDims) && positiveWavDims(1) == 1
        if ~isempty(CelWavCoeffs{1,j})
            MatWavCoeffs{j}(1:cumSzBlocks(1),1:BlockWavDims_j(1)) = CelWavCoeffs{1,j};
        end
        MatWavDims(1:cumSzBlocks(1),j) = BlockWavDims_j(1);
        positiveWavDims = positiveWavDims(2:end);
    end
    
    for n = positiveWavDims % n>1
        if ~isempty(CelWavCoeffs{n,j})
            MatWavCoeffs{j}(cumSzBlocks(n-1)+1:cumSzBlocks(n),1:BlockWavDims_j(n)) = CelWavCoeffs{n,j};
        end
        MatWavDims(cumSzBlocks(n-1)+1:cumSzBlocks(n),j) = BlockWavDims_j(n);
    end
    
end

MatWavCoeffs = [MatWavCoeffs{:}]; % matrix of wavelet coefficients, element by scale

return;

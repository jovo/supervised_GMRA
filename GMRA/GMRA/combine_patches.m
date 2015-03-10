function I = combine_patches(P, imageSizes, overlapping)

% The rows of P are the patches, which are assumed to be square.

if nargin<3,
    overlapping = 0;
end

if numel(imageSizes)==1
    imageSizes = [imageSizes imageSizes];
end

m = imageSizes(1);
n = imageSizes(2);

[~, patchSize2] = size(P); 
patchSize = sqrt(patchSize2);        
Plong = reshape(P', patchSize, []);

I = zeros(m,n);
switch overlapping
    
    case 0 % nonoverlapping
        
        nPatches_perCol  = m/patchSize;
        nPatches_perRow = n/patchSize;
            
        for i = 1:nPatches_perCol
            I((i-1)*patchSize+1:i*patchSize, :) = Plong(:, repmat((i-1)*patchSize+1:i*patchSize, 1, nPatches_perRow)+ m*reshape(repmat(0:nPatches_perRow-1, patchSize,1),1,[]));
        end
    
    %case patchSize/2  
        
    otherwise 
        
        spacing = patchSize-overlapping; % assumed to be a common divisor of m and n   
        weight = spacing^2/(m*n);
       
        patchRowIndex = 1:spacing:m+1-patchSize;
        patchColIndex = 1:spacing:n+1-patchSize; %both relative to original mxn image
        
        for j = 1:length(patchColIndex)
            currentImageCols = patchColIndex(j):patchColIndex(j)+patchSize-1;
            for i = 1:length(patchRowIndex)
                currentImageRows = patchRowIndex(i):patchRowIndex(i)+patchSize-1;
                I(currentImageRows, currentImageCols) = I(currentImageRows, currentImageCols) + weight*Plong(:, ((j-1)*length(patchRowIndex)+(i-1))*patchSize+(1:patchSize));
            end
        end
        
end




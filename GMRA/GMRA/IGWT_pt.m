function [x,x_mat] = IGWT_pt(gW, Data)

% Inverse Geometric Wavelet Transform
%
% Input: 
%   gW: structure of wavelet bases and translations
%     .WavBases: wavelet basis
%     .WavConsts: associated translations
%   Data: structure of wavelet coefficients
%     .WavCoeffs: wavelet coefficients
%     .chain: the path up the tree
%
% Output:
%   x_mat: matrix, the jth row contains the recovered coordinates at scale j
%   x: = x_mat(end,:) is the recovered data at finest scale 

j_max = length(Data.chain); % number of scales involved

x_mat = [gW.WavConsts{Data.chain}]; % matrix of recovered coordinates at all scales, initialized as the wavelet constant at each scale

for j = j_max:-1:1
    
    if ~isempty(Data.WavCoeffs{j})
        x_mat(:,j) = x_mat(:,j) + gW.WavBases{Data.chain(j)}*Data.WavCoeffs{j}';
    end
    
end

if gW.opts.addTangentialCorrections
    
    for j = j_max-1:-1:2
        x_mat(:,j) = x_mat(:,j) - gW.ScalFuns{Data.chain(j-1)}*(gW.ScalFuns{Data.chain(j-1)}'*sum(x_mat(:, j+1:j_max),2));
    end
    
end

x_mat = cumsum(x_mat,2);
x = x_mat(:,end);

return;

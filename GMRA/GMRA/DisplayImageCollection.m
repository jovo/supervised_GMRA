function matI = DisplayImageCollection(I,maxCols,linewidth,display)

if nargin<4; display = 1; end;
if nargin<3; linewidth = 1; end;
if nargin<2 || isempty(maxCols); maxCols  = Inf; end;

[M,N,L] = size(I); % L images of size MxN

nCols = min(round(sqrt(1.7786*L)), maxCols); % number of images in a row
nRows = ceil(L/nCols); % number of images in a column
if L<=2,
    nCols = L;
    nRows = 1;
end;

matI = repmat(min(min(min(I))), [M*nRows+linewidth*(nRows-1), N*nCols+linewidth*(nCols-1)]); % big image matrix
for i = 1:nRows-1
    for j = 1:nCols
        matI((i-1)*(M+linewidth)+1:i*M+(i-1)*linewidth, (j-1)*(N+linewidth)+1:j*N+(j-1)*linewidth) = I(:,:,(i-1)*nCols+j);
    end
end

% i = nRows; % last row
for j = 1:L-(nRows-1)*nCols
    matI((nRows-1)*(M+linewidth)+1 : nRows*M+(nRows-1)*linewidth, (j-1)*(N+linewidth)+1 : j*N+(j-1)*linewidth) = I(:,:,(nRows-1)*nCols+j);
end

if nargin>3
    if display,
        figure; imagesc(matI); colormap gray
    end;
end;

return;
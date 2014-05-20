function GWT_DisplayCoeffs( GWT, Data )

%
% Displays tables of (1) magnitude of wavelet coefficients, (2) wavelet coefficients, (3) dimensions of the wavelet subspaces
%

J = max(GWT.Scales);
N = size(GWT.X,2);

%% form the matrix of wavelet coefficients 
WavCoeffs = Data.MatWavCoeffs';
WavCoeffs2 = WavCoeffs.^2; 

WavCoeffMags = zeros(J,N);
for j = 1:J
   WavCoeffMags(j,:) = sqrt(sum(WavCoeffs2(sum(Data.maxWavDims(1:j-1))+1:sum(Data.maxWavDims(1:j)),:),1));
end

%% Plots

DrawGWTWavSubFcn( GWT, log10(WavCoeffMags), 'Magnitude of Wavelet Coefficients (log10 scale)' );
DrawGWTCoeffs( GWT, WavCoeffs, Data.maxWavDims, 'Wavelet Coefficients' );
DrawGWTWavSubFcn( GWT, Data.MatWavDims', 'Dimension of wavelet subspaces' );


%% coefficients against scales
MeanWavCoeffMags = mean(WavCoeffMags, 2);

delta = zeros(1,J);
for j = 1:J
    delta(j) = mean(GWT.Radii(GWT.Scales == j));
end;

figure; 
plot(-log10(delta), log10(MeanWavCoeffMags), '*', 'MarkerSize',10); 
title('coefficients against scale (log10 scale)', 'fontSize', 12)
xlabel('scale', 'fontSize', 12); ylabel('coefficient', 'fontSize', 12)
grid on
if J>3 
    PlotPolyFit( -log10(delta(2:end-1)'), log10(MeanWavCoeffMags(2:end-1)) ); 
elseif J>1
    PlotPolyFit( -log10(delta'), log10(MeanWavCoeffMags) );
end;
axis equal;

return;


function DrawGWTCoeffs( GWT, Coeffs, maxWavDims, Title )

if nargin<4, Title =''; end;

J = max(GWT.Scales);

%% Show the matrix of all wavelet coefficients
figure; imagesc(Coeffs); 
cmap=map2;
colormap(cmap); 
title(Title, 'fontSize', 12);
xlabel('points', 'fontSize', 12); ylabel('scales', 'fontSize', 12);
linewdth = 1;
col = 0.8*ones(1,3);

for j = 1:J
    nets = sort([find(GWT.Scales == j) GWT.LeafNodes(GWT.Scales(GWT.LeafNodes)<j)], 'ascend');
    lens  = cumsum(GWT.Sizes(nets)); 
    ln = line([0 lens(end)], [sum(maxWavDims(1:j-1))+0.5 sum(maxWavDims(1:j-1))+0.5]); set(ln,'color',col,'LineWidth',linewdth);
    for i = 1:length(lens)-1
        ln = line([lens(i) lens(i)], [sum(maxWavDims(1:j-1))+0.5 sum(maxWavDims(1:J))+0.5]); set(ln,'color',col,'LineWidth',linewdth);
    end
end
balanceColor
colorbar

return;


function DrawGWTWavSubFcn( GMRA, Fcn, Title )

%
% Displays a function of the wavelet subspaces in an image structured as the wavelet tree.
%

if nargin<3, Title =''; end;

J = max(GMRA.Scales);
leafNodes = GMRA.LeafNodes;

figure;
imagesc(Fcn);
if(min(size(Fcn)))==1,
    cmap=map2; %copper;%map2;
    cmap(1,:)=[1,1,1];
    colormap(cmap);
end;
title(Title, 'fontSize', 12)
xlabel('points', 'fontSize', 12); ylabel('scales', 'fontSize', 12);

if false,
linewdth = 1;
col = 0.8*ones(1,3);

for j = 1:J
    nets = sort([find(GWT.Scales == j) leafNodes(GWT.Scales(leafNodes)<j)], 'ascend');
    lens  = cumsum(GWT.Sizes(nets)); 
    ln = line([0 lens(end)], [(j-1)+0.5 (j-1)+0.5]); set(ln,'color',col,'LineWidth',linewdth);
    for i = 1:length(lens)-1
        ln = line([lens(i) lens(i)], [(j-1)+0.5 J+0.5]); set(ln,'color',col,'LineWidth',linewdth);
    end
end

end;
%balanceColor
colorbar

return;
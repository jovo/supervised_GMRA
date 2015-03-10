function GWT_DisplayDataAndPlanes( gW, j )

figure;
scatter3( gW.X(:,1),gW.X(:,2), gW.X(:,3), 5 );

kIdxs = find(gW.Scales==j);

for i = 1:length(kIdxs),
    lCenter = gW.Centers{kIdxs(i)};
    lRadius = gW.Radii(kIdxs(i));
    lPlane  = gW.ScalFuns{kIdxs(i)};
    
    drawpatch( lCenter,lRadius,lPlane(1:3,:) );
end;

return;

function drawpatch( center, radius, plane )

xyz = [1,1,0

patch([center(1)+radius/2*plane(1,1), center(1)+radius/2*plane(1,1),center(1)-radius/2*plane(1,1), center(1)-radius/2*plane(1,1)],[center(2)+radius/2*plane(1,1), center(1)+radius/2*plane(1,1),center(1)-radius/2*plane(1,1), center(1)-radius/2*plane(1,1)]

return;
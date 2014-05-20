function [hatx,planeidxs,l0norm,l1norm] = IGWT_CS( gMRA, XProj, P, j )

%
% function IGWT_CS( gMRA, Xproj, P, j )
%
% IN:
%   gMRA        : GMRA structure
%   XProj       : Px, where P is a random projector and x the points to be reconstructed
%   P           : projector used to project the x's onto Xproj
%   j           : scale at which to reconstruct
%
% OUT:
%   hatx        : reconstructed approximation to x's
%   planeidxs   : indices into gMRA.ScalFuns for the planes onto which the x's are reconstructed
%   l0norm      : mean l_0 norm of the coefficients of hatx
%   l1norm      : mean l_1 norm of the coefficients of hatx
%
%
%
% EXAMPLE:
%
%

% (c) Copyright Duke University,
% Mauro Maggioni and Mark Iwen
% mauro@math.duke.edu

% Get the centers at scale j
PlaneIdxs      = get_partition_at_scale(gMRA,j);
Centers        = [gMRA.Centers{PlaneIdxs}];
CentersProj    = P*Centers;

[~, ClosestIdx, Dists] = nrsearch(CentersProj, XProj, 1, 0, [], struct('ReturnAsArrays',1));

lUniqueClosestIdxs = unique(ClosestIdx);
for k = 1:length(lUniqueClosestIdxs),
    idxs{k} = find(ClosestIdx==lUniqueClosestIdxs(k));
    Basis{k} = gMRA.ScalFuns{PlaneIdxs(lUniqueClosestIdxs(k))}; 
    CentersProj_tmp{k} = CentersProj(:,lUniqueClosestIdxs(k));
    Centers_tmp{k} = Centers(:,lUniqueClosestIdxs(k));
end;

hatx_tmp = cell(length(lUniqueClosestIdxs),1);
l0norm = 0;
l1norm = 0;
for k = 1:length(lUniqueClosestIdxs),   
    u = (P*Basis{k})\(bsxfun(@minus,XProj(:,idxs{k}),CentersProj_tmp{k}));
    
    l0norm = l0norm + length(find(u(:)~=0));
    l1norm = l1norm + sum(abs(u(:)));
    
    hatx_tmp{k} = bsxfun(@plus,Basis{k}*u,Centers_tmp{k});
end;

hatx = zeros(size(P,2),size(XProj,2));
for k = 1:length(lUniqueClosestIdxs),
    hatx(:,idxs{k}) = hatx_tmp{k};    
end;

planeidxs = lUniqueClosestIdxs;
l0norm    = l0norm/size(XProj,2);
l1norm    = l1norm/size(XProj,2);

return;


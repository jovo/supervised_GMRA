function [XProj,P,distortion] = RandomProject( X, d)

[P,R] = qr(randn(size(X,1),d));
if d<=size(P,1),
    P     = P(1:d,:)*sqrt(size(X,1)/d);
else
    P     = P*sqrt(size(X,1)/d);
end;

XProj = P*X;

if nargout>2,
    DistOrig = pdist(X');
    DistProj = pdist(XProj');
    distortion = max(DistProj./DistOrig)/min(DistProj./DistOrig);
end;

return;
function dist = distancessq( X, Y )

dist = zeros(size(X,2),size(Y,2));

for k = 1:size(X,2),
    dist(k,:) = sum(bsxfun(@minus,Y,X(:,k)).^2,1);
end;


return;

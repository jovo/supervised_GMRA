function s = rownorms( X,p )

if p<inf,
    s = sum(abs(X).^p,2).^(1/p);
else
    s = max(abs(X),[],2);
end;

return;
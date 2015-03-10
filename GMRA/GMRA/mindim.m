function dim = mindim(sigs, errorType, err)

% This function finds the minimum dimension of the node,
% using the local singular values,
% so that the averaged L2 error is below the given precision 
%
% Variables:
% sigs: vector of singular values, in decreasing order
% errorType: 'absolute' or 'relative'
% err: precision (must include the factor sqrt(N))

s2 = sum(sigs.^2);

if strcmpi(errorType, 'absolute')
    tol = err^2;
else % relative error
    tol = err*s2;
end

dim = 0;
while dim<length(sigs) && s2>tol
    dim = dim+1;
    s2 = s2 - sigs(dim)^2;
end

% s2 = sigs.^2;
% cums2 = cumsum(s2(end:-1:1));
% cums2 = cums2(end:-1:1);
% dim = find(cums2<err, 1, 'first') - 1;

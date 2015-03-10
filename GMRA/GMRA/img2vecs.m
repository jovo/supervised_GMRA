function A = img2vecs(X,nl)

% Construct a matrix A such that each column is a subimage of the
% original image.
% nl - length of cube for each subimage.
% X - given image

Nsub = nl^2;   % Number of pixels in each sub_image

szx = size(X);

szA2 = (szx(1)-nl+1)*(szx(2)-nl+1);

A = zeros(szA2,Nsub);

k = 1;
for m = 1:nl
   for n = 1:nl
      A(:,k) = reshape(X(n:n+szx(1)-nl,m:m+szx(2)-nl), szA2, 1);
      k = k+1;
   end
end
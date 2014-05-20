function h = map2(m)
%HOT2    Black-red-yellow-white color map.
%   HOT(M) returns an M-by-3 matrix containing a "hot" colormap.
%   HOT, by itself, is the same length as the current colormap.
%
%   For example, to reset the colormap of the current figure:
%
%             colormap(hot)
%
%   See also HSV, GRAY, PINK, COOL, BONE, COPPER, FLAG, 
%   COLORMAP, RGBPLOT.

%   C. Moler, 8-17-88, 5-11-91, 8-19-92.
%   Copyright 1984-2001 The MathWorks, Inc. 
%   $Revision: 5.6 $  $Date: 2001/04/15 11:58:57 $

if nargin < 1, m = size(get(gcf,'colormap'),1); end
n = fix(1/2*m);
ramp = (n:-1:1)'/(n);

b = [ones(n,1); (0.75*ramp+0.25)];
g = [0.5*(1-ramp)+0.5; 0.5*ramp+0.5];
r = [0.75*(1-ramp)+0.25; ones(n,1)];

h = [r g b];

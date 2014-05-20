function H = entropy( v )

v = v + (v==0);
H = -1 * sum(v .* log2(v));

return;

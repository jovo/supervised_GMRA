function y = f_displayenergystdcoord( GWT, n, Data )

if GWT.Scales(n)>0,
    lF = sum(GWT.WavBases{n}.^2,2);
else
    lF = sum(GWT.ScalFuns{n}.^2,2);
end;

y = std(find(lF>0.5*max(lF)));

return;
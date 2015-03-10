function y = f_displayenergycoord( GWT, n, Data )

if GWT.Scales(n)>0,
    lF = sum(GWT.WavBases{n}.^2,2);
else
    lF = sum(GWT.ScalFuns{n}.^2,2);
end;

y = mean(find(lF>0.8*max(lF)));

return;
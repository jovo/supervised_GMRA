function balanceColor( scaleFactor )
% balanceColor -- balance colormap with zero at center

if nargin < 1,
    scaleFactor = 1.0;
end	

[cmin cmax] = caxis;

if(cmax>=0 && cmin<=0)
    if(abs(cmax) >= abs(cmin))
        caxis([-1.0*scaleFactor*cmax scaleFactor*cmax]);
    else
        caxis([scaleFactor*cmin -1.0*scaleFactor*cmin]);
    end
    
elseif(cmax>=0 && cmin>=0)
        caxis([-1.0*scaleFactor*cmax scaleFactor*cmax]);
    
elseif(cmax<=0 && cmin<=0)
        caxis([scaleFactor*cmin -1.0*scaleFactor*cmin]);
    
end;
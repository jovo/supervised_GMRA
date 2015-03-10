function [Costs,CoeffCosts] = GWT_DisplayCost( GWT, Data, EpsilonRange )

CoeffCosts  = zeros(length(EpsilonRange),1);

WavMags = abs(Data.MatWavCoeffs);

for k = 1:length(EpsilonRange),
    thresCoeffs     = WavMags>EpsilonRange(k);
    if sum(sum(thresCoeffs,1)>0),
        fprintf('!');
    end;
    CoeffCosts(k)   = sum(sum(thresCoeffs));
end;

Costs = CoeffCosts + GWT.DictCosts;

figure;plot(log10(EpsilonRange),CoeffCosts,'-x');title('Cost of encoding coefficients as function of threshold'); ylabel('Coeff. Cost'); xlabel('Threshold');
figure;plot(log10(EpsilonRange),Costs,'-x');title('Total cost as function of threshold'); ylabel('Total Cost'); xlabel('Threshold');
hold on;plot(log10(EpsilonRange),numel(GWT.X),'r-o');

return;
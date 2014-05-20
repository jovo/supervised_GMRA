function [Y,GWT_Y] = GWT_GenerateRandomPts( GWT, TrainingData, opts )

% Pick a random leaf, according to the empirical distribution of points in leaves
p       = histc(GWT.IniLabels,1:length(GWT.LeafNodes));     %  Empirical parameters for multinomial distribution
p       = p/sum(p);
iniLeaf = find(mnrnd(1,p));
iniLeaf = 1;

% Find the statistics of the multiscale coefficients for the points in this leaf
wavcoeffs=TrainingData.MatWavCoeffs(find(GWT.IniLabels==iniLeaf),:);
coeff_mean  = mean(wavcoeffs);
coeff_cov   = cov(wavcoeffs);
[~,S,V]     = svd(coeff_cov);
wavcoeffs   = coeff_mean+randn(1,size(S,1))*S*V';
%wavcoeffs   = TrainingData.MatWavCoeffs(find(GWT.IniLabels==iniLeaf),:);
%wavcoeffs = wavcoeffs(1,:);
%wavcoeffs = zeros(size(wavcoeffs));

% Get the path from the leaf to the root
chain = fliplr([iniLeaf,get_ancestors(GWT.cp,iniLeaf)]);

idx = 1;
for i = 1:length(chain),
    WavCoeffs{i} = wavcoeffs(idx:idx+size(GWT.WavBases{chain(i)},2)-1);
    idx = idx+size(GWT.WavBases{chain(i)},2);
end;
% 
% % Running up the chain, draw random coefficients
% for i = 1:length(chain),
%     % Estimate the covariance matrix
%     ltmp         = size(GWT.WavBases{chain(i)},2);
%     WavCoeffs{i} = 0.1*(GWT.WavSingVals{chain(i)}(1:ltmp).*randn(ltmp,1))'; % nonsense? +mean(TrainingData.CelWavCoeffs{iniLeaf,get_scale(GWT.cp,chain(i))});
% end;

% This is the GWT transform of the point to be generated
GWT_Y.chain     = chain;
GWT_Y.WavCoeffs = WavCoeffs;

% Transform back to get the new point
Y = IGWT(GWT,GWT_Y);

return;
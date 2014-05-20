function [GWTstats, SVDstats, thresSVDstats, gW, Data] = GenerateRateDistortionCurve( pExampleIdx, pGWTversion, thresholds )

%% Generate Data and Set Parameters
[X, GWTopts] = GenerateData_and_SetParameters(pExampleIdx);
[D,N] = size(X);

%% compute svd costs and errors for all possible choices of dimension
if nargout>1
    lMean = mean(X,2);
    Xcentered = X-repmat(lMean,1,N);
    [U,S,V] = svd(Xcentered,0);
    SVDcoeffs = S*V';
    s = diag(S);
    Scumsum = cumsum(s.^2);
    
    SVDstats.X = X;
    SVDstats.dims = 1:length(s);
    SVDstats.coeffsCosts = SVDstats.dims;
    SVDstats.dictCosts = (SVDstats.dims+1);
    SVDstats.err_X = sqrt((Scumsum(end)-Scumsum)/N);
end

%% Construct geometric wavelets and compute wavelet coefficients using the intial precision
GWTopts.GWTversion = pGWTversion;
gW = GMRA(X,GWTopts);
Data = FGWT_trainingData(gW, X);

%%
ntrials = length(thresholds);

GWTstats.dictCosts   = zeros(1,ntrials);
GWTstats.coeffsCosts  = zeros(1,ntrials);
GWTstats.err_X = zeros(1,ntrials);
GWTstats.err_Xclean = zeros(1,ntrials);

if nargout>1
    SVDstats.lSVDidxthres = zeros(1,ntrials);
    thresSVDstats = GWTstats;
    thresSVDstats.lSVDidxthres = SVDstats.lSVDidxthres;
end

for i = 1:ntrials
    
    if nargout>1
        SVDcoeffs1 = SVDcoeffs;
        SVDcoeffs1(abs(SVDcoeffs1)<thresholds(i)) = 0;
        thresSVDstats.coeffsCosts(i) = nnz(SVDcoeffs1)/N;
        
        nzRows = (mean(SVDcoeffs1.^2,2)>1e-12);
        thresSVDstats.lSVDidxthres(i) = sum(nzRows);
        thresSVDstats.dictCosts(i) = thresSVDstats.lSVDidxthres(i); 
        
        SVDapprox = U(:,nzRows)*SVDcoeffs1(nzRows,:);
        thresSVDstats.err_X(i) = GWT_ApproxError( Xcentered, SVDapprox, struct('norm', 2, 'relative', false) );
    end
    
    if GWTopts.GWTversion<3 % regular or orthogonal GWT
        Data1 = Data;
        Data1.CelWavCoeffs(:,1:end) = threshold_coefficients(Data.CelWavCoeffs(:,1:end), struct('shrinkage', 'hard', 'coeffs_threshold', thresholds(i)));
        [gW1, Data1]  = simplify_the_GWT_tree(gW, Data1);
    else % pruning GWT
        GWTopts.precision = thresholds(i);
        gW1 = GMRA(X,GWTopts);
        Data1 = FGWT_trainingData(gW1, X);
    end
    
    GWTstats.coeffsCosts(i) = Data1.CoeffsCosts/N;
    GWTstats.dictCosts(i)  = gW1.DictCosts/D;
    
    Data1.Projections = IGWT_trainingData(gW1, Data1.CelWavCoeffs);
    GWTstats.err_X(i) = GWT_ApproxError( X, Data1.Projections(:,:,end), struct('norm', 2, 'relative', false) );
    if isfield(GWTopts, 'X_clean')
        GWTstats.err_Xclean(i)   = GWT_ApproxError( GWTopts.X_clean, Data1.Projections(:,:,end), struct('norm', 2, 'relative', false) );    
    end
    
    if nargout>1
        SVDstats.lSVDidxthres(i) = round((GWTstats.coeffsCosts(i)+GWTstats.dictCosts(i))/(N+D));
    end
    
end

%%
GWTstats.totalCosts = GWTstats.coeffsCosts*N+GWTstats.dictCosts*D;

if nargout>1
    SVDstats.totalCosts = SVDstats.coeffsCosts*N+SVDstats.dictCosts*D;
    thresSVDstats.totalCosts = thresSVDstats.coeffsCosts*N+thresSVDstats.dictCosts*D;
end

return

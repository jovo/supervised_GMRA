function gMRA = GMRA(X,opts,Y)

%% Geometric MultiResolution Analysis (GMRA) for Data Sets
%
% Usage: gMRA = GMRA(X, opts)
% 
% INPUT: 
%   X: D-by-N data matrix with each column being a data point
%   opts: a structure with the following optional parameters:
%       [knn]              : size of the neighborhood graph. Default: 50.
%       [knnAutotune]      : local scaling parameter. Default: 20.
%       [ManifoldDimension]: if a positive number, then it will be provided to all nodes as local dimension, 
%                            except for the leaf nodes whose dimensions are determined separately (see below). 
%                            If zero (default), then the local dimensions at all nodes are determined adapatively 
%                            by the following fields:
%       [errorType]      : 'relative' or 'absolute' error computed from the local singular values. Default: 'relative'.
%       [smallestMetisNet] : lower bound for the sizes of the METIS nets. Default: 10.
%       [threshold0]     : corresponding threshold to each error type, can be a scalar or 
%                             a vector indexed from coarsest scale (1) to finest scale (J) .
%                            (if length < J, the last element will be replicated for scales (J-length+1):J).   
%                            Default: 0.5 (uniformly for all J scales).
%       [precision]        : precision parameter for the leaf nodes; if provided, it will overwrite threshold0 at the leaf nodes. 
%                            Default = threshold0.
%       [threshold1]       : threshold for determining the wavelet dimensions. 
%                            Default: 1e-5.
%       [addTangentialCorrections]: whether to add tangential corrections so as to use the best approximations at each scale.
%                            Default: true.
%       [avoidLeafnodePhi] : whether to avoid storing the scaling functions at finest scale (and use instead 
%                            the span of the parent scaling function and the children wavelet functions). 
%                            Default: true.
%       [sparsifying]      : whether to sparsify the wavelet bases using algorithms such as k-svd and SPAMS.
%                            Default: false.
%           [sparsifying_method]:'ksvd' (by M. Elad et al.) or 'spams' (by J. Mairal). Default: 'ksvd'.
%           [sparsifying_oversampling] : when sparsifying, will seek a dictionary of size (sparsifying_oversampling)*(dimension of wavelet subspace). Default: 2.
%       [splitting]        : whether to separate out the common intersection of the wavelet subspaces 
%                            associated to the children of a fixed node in order to save the encoding cost.
%                            Default: false. 
%       [threshold2]       : threshold for determining the intersection dimension of the wavelet subspaces associated to all children of a fixed node.
%                            Default: 0.1.
%       [mergePsiCapIntoPhi]:whether to merge the common part of the wavelet subspaces associated to the children into the scaling function of the parent.
%                            Default: false. 
%       [GWTversion]       : one of the following:
%                               =0, plain construction
%                               =1, orthogonal geometric wavelets
%                               =2, minimal encoding-cost pruning
%                            Default: 1.
%       [shrinkage]        : 'soft' or 'hard' shrinkage of the wavelet coefficients. Default: 'hard'.
%       [coeffs_threshold] : threshold for shrinking wavelet coefficients. Default: 1e-10. 
%       [verbose]          : level of verbosity, a number in {0,1}. Default: 0.
%       [MaxDim]           : maximum subspace dimension to be ever considered. Default:100.
%       [graph]            : graph structure as returned by GraphDiffusion. If provided, does not recompute the graph.
%       [tree]             : tree information as returned by nesdis (nested dissection). It will be gMRA.cp as described below. 
%                            If provided, does not rerun nesdis.
%       [treeidxs]         : same as cmembers returned by nesdis. If provided, does not rerun nesdis.
%
% OUTPUT:
%   gMRA: a structure with the following fields:
%       .cp                : vector encoding the metis tree structure with .cp(x) = the parent of x. Each node is a subset of X and a parent is the union of its children
%       .LeafNodes         : vector of labels of the leaf nodes.
%       .isaleaf           : vector of 1 (leaf node) or 0 (non-leaf node).
%       .IniLabels         : the labels of the data points w.r.t. the leaf nodes in the METIS tree (all the leaves are a full partition of the data).
%       .PointsInNet       : cell array, PointsInNet{i} contains the labels of all the points in the node i of the metis tree.
%       .Sizes             : vector of number of points in each node.
%       .Radii             : vector of the radius of each node.
%       .Scales            : vector of scales of all the nodes in the metis tree; the root has scale 1 and a larger scale implies a finer approximation.
%       .Centers           : cell array of the centers of the nodes.
%       .ScalFuns          : cell array of the local bases of the nodes.
%       .Sigmas            : cell array of the local singular values 
%       .WavBasis          : cell array of wavelet bases; each cell encodes the basis vectors that are present in the current node but orthogonal to the
%                               parent. For simplicity, the wavelet basis at the root is identified with the scaling function at the root.
%       .WavConsts         : cell array of wavelet translations that are needed to move to a node from its parent.
%       .WavSingVals       : cell array of corresponding singular values associated with the wavelet bases. At the root, it coincides with .Sigmas
%       .epsEncodingCosts  : vector of total encoding costs at each node when approximating the local data by the lowest-dimensional pca plane within the given precision .threshold0.
%       .DictCosts         : overall cost for storing the dictionary (i.e., wavelet bases and constants). When .addTangentialCorrections = true, the scaling functions will also be included in the dictionary.
%       .Timing            : structure with the following fields:
%                               graph  : cputime (in seconds) taken by the graph construction
%                               nesdis : cputime (in seconds) taken by multiscale partitioning
%                               GW     : cputime (in seconds) for the whole construction
%
% Required Packages:
%   1. Diffusion Geometry [by Mauro Maggioni]
%   2. Metis              [by George Karypis et al.]
%   3. SuiteSparse        [by Tim Davis, for the metis and nesdis wrappers]
%   4. LightSpeed         [by Tom Minka]
% If sparsifying wavelet basis, then also need the following two packages:
%   5. K_SVD              [by Michael Elad et al.] and/or
%   6. SPAMS              [by Julien Mairal et al.]
% 
% Publications:
%   1. Multiscale Geometric Methods for Data Sets II: Geometric Wavelets, W.K. Allard, G. Chen and M. Maggioni, ACHA, 2011 
%   2. Multiscale Geometric Dictionaries for Point-Cloud Data, G. Chen, and M. Maggioni, The 9th International Conference on Sampling Theory and Applications (SampTA), Singapore, 2011
%   3. Multiscale Geometric Wavelets for the Analysis of Point Clouds, G. Chen and M. Maggioni, The 44th Annual Conference on Information Sciences and Systems (CISS), Princeton, NJ, 2010
%
% (c) 2011 Mauro Maggioni and Guangliang Chen, Duke University
% Contact: {mauro, glchen}@math.duke.edu

%% Parameters
gMRA.Timing.GW = cputime;

if nargin<1,            
    error('The input X must be provided.');  
end;

if nargin<2,           
    opts = struct();
end;

if nargin<3,
    Y = [];
end

opts.AmbientDimension  = size(X,1);

if ~isfield(opts, 'GWTversion') || isempty(opts.GWTversion),  
    opts.GWTversion = 1; 
end;
opts.orthogonalizing  = (opts.GWTversion == 1)
opts.pruning          = (opts.GWTversion == 2)

% parameters for building neighborhood graph
if ~isfield(opts, 'knn'),                opts.knn = 50;                end;
if ~isfield(opts, 'knnAutotune'),        opts.knnAutotune = 30;        end;
if ~isfield(opts, 'smallestMetisNet'),   opts.smallestMetisNet = 30;   end;

% parameters for choosing local PCA dimensions
if ~isfield(opts, 'ManifoldDimension'),    
    opts.ManifoldDimension = 0;                 
elseif opts.pruning && opts.ManifoldDimension>0, % conflict
    opts.ManifoldDimension = 0;
    warning('Manifold Dimension is NOT used and has been set to zero in order to allow for locally adaptive dimensions!'); %#ok<WNTAG>
end;

if ~isfield(opts, 'threshold0'),        opts.threshold0 = 0.5;         end;                    
if ~isfield(opts, 'errorType'),         opts.errorType = 'relative';   end;                    
if ~isfield(opts, 'precision'),         opts.precision = 0.05;         end;
opts.GWTversion
switch opts.GWTversion
    case 0
        if ~isfield(opts, 'addTangentialCorrections'),  opts.addTangentialCorrections = false; end;
        if ~isfield(opts, 'splitting'),                 opts.splitting = false;                end;
        if ~isfield(opts, 'sparsifying'),               opts.sparsifying = false;              end;
        if ~isfield(opts, 'sparsifying_oversampling'),  opts.sparsifying_oversampling = 2;     end;
        if opts.sparsifying,                            opts.avoidLeafnodePhi = false; 
        elseif ~isfield(opts, 'avoidLeafnodePhi'),      opts.avoidLeafnodePhi = true;          end;
        if opts.sparsifying && ~isfield(opts, 'sparsifying_method')
            opts.sparsifying_method = 'ksvd';
        end
    case {1,2}
        opts.addTangentialCorrections = false;
        opts.avoidLeafnodePhi = false;
        opts.splitting = false;
            opts.sparsifying = false;
        if opts.GWTversion == 2,
            opts.sparsifying = false;
        end;
end

% parameter for choosing wavelet dimensions
if ~isfield(opts, 'threshold1'),           
    opts.threshold1 = 1e-5;                     
end;
if (opts.splitting || opts.pruning) && ~isfield(opts, 'threshold2'),     
    opts.threshold2 = 1e-1;        
end;
if opts.splitting && ~isfield(opts, 'mergePsiCapIntoPhi'), 
    opts.mergePsiCapIntoPhi = false; 
end;

if ~isfield(opts, 'shrinkage'),     
    opts.shrinkage = 'hard';    
end;
if ~isfield(opts, 'coeffs_threshold'),                  
    opts.coeffs_threshold = 1e-10;   
end;

if ~isfield(opts, 'verbose'),       opts.verbose = 0;                  end;
if ~isfield(opts, 'MaxDim'),        opts.MaxDim = 100;                 end;
if ~isfield(opts, 'graph'),         opts.graph = [];                   end;
if ~isfield(opts, 'tree'),          opts.tree = [];                    end;
if ~isfield(opts, 'treeidxs'),      opts.treeidxs = [];                end;

%%
gMRA.X = X;
gMRA.opts = opts;
gMRA.Y = Y;

%% Compute pairwise weights
if isempty(opts.graph),
    if gMRA.opts.verbose,     fprintf('\n Constructing graph...'); end; %tic                     
    gMRA.Timing.graph = cputime;
    gMRA.Graph = GraphDiffusion( X, 0, ...
        struct('KNN', opts.knn,'kNNAutotune',opts.knnAutotune,'Display',0,...
               'kEigenVecs',5,'Symmetrization','W+Wt','NNMaxDim',100) );
    gMRA.Timing.graph = cputime-gMRA.Timing.graph;
    if gMRA.opts.verbose,     fprintf('done. (%.3f sec)',toc); end;
end;

%% Build the metis tree and remove separators
if isempty(opts.tree) || isempty(opts.treeidxs),
    if gMRA.opts.verbose,   fprintf('\n Constructing multiscale partitions...'); end; %tic
    gMRA.Timing.nesdis    = cputime;
    [~,gMRA.cp,cmember]   = nesdis(gMRA.Graph.W,'sym',opts.smallestMetisNet);
    gMRA.Timing.nesdis    = cputime-gMRA.Timing.nesdis;
    if gMRA.opts.verbose,   fprintf('done. (%.3f sec)',toc); end;
else
    gMRA.cp = opts.tree;
    %cmember = opts.treeidxs;
end;
%
gMRA.Scales     = compute_scales(gMRA.cp);
gMRA.isaleaf    = leafnodes(gMRA.cp);
gMRA.LeafNodes  = find(gMRA.isaleaf);
gMRA.IniLabels  = dissolve_separator(X, gMRA.cp, cmember, gMRA.LeafNodes, 'centers');

%% Initialize GWT structure
nAllNets        = length(gMRA.cp);
inputThreshold0 = gMRA.opts.threshold0; 
J               = max(gMRA.Scales);
J0              = numel(inputThreshold0);
if J0 < J % too short, the use last element for finer scales
    inputThreshold0 = [inputThreshold0 repmat(inputThreshold0(end), 1, J-J0)];
end
% The following line converts threshold0 from per scale (1 by J vector) to per node (1 by nAllNets vector):
gMRA.opts.threshold0 = inputThreshold0(gMRA.Scales);                % node-dependent precision

if gMRA.opts.precision<min(gMRA.opts.threshold0)      % separate thresholds for leaf nodes are provided
    if gMRA.opts.pruning % minimal encoding cost pruning
        gMRA.opts.threshold0(1:end) = gMRA.opts.precision;
    else
        gMRA.opts.threshold0(gMRA.LeafNodes) = gMRA.opts.precision;   % overwrite the leaf nodes with the specified precision
    end
end

% Initialize fields of gMRA
gMRA.Radii        = zeros(1,nAllNets);
gMRA.Sizes        = zeros(1,nAllNets);
gMRA.PointsInNet  = cell(1, nAllNets);
gMRA.Centers      = cell(1,nAllNets);
gMRA.ScalFuns     = cell(1,nAllNets);
gMRA.Sigmas       = cell(1,nAllNets);
gMRA.WavBases     = cell(1,nAllNets);
gMRA.WavConsts    = cell(1,nAllNets);
gMRA.WavSingVals  = cell(1,nAllNets);

if gMRA.opts.sparsifying,
    gMRA.Projections = cell(1,nAllNets); % coordinates of the projected data to each node 
end

if gMRA.opts.splitting
    gMRA.WavDimsPsiCap = zeros(1,nAllNets); % intersection dimensions of the wavelet subspaces associated to the children of a specific node
end
    
% back up gMRA.cp
gMRA.cp_orig = gMRA.cp;

%% Construct geometric wavelets
if gMRA.opts.verbose,     fprintf('\n Constructing geometric wavelets...'); end; %tic
if gMRA.opts.orthogonalizing
    gMRA = construct_orthogonalGeometricWavelets( gMRA );           % orthogonal geometric wavelets
elseif gMRA.opts.pruning
    gMRA = construct_pruningGeometricWavelets( gMRA );               % minimal encoding-cost pruning
else
    gMRA = construct_GMRA( gMRA );                                           % best approximations, no pruning          
end
if gMRA.opts.verbose,     fprintf('done. (%.3f sec) \n',toc); end; 

%% Compute cost of dictionary
gMRA.DictCosts = computeDictionaryCost(gMRA);

%%
gMRA.Timing.GW = cputime-gMRA.Timing.GW;

fprintf('debugging 2014-06-26 \n')
return;

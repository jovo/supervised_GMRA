function gW = sparsifying_WaveletBases(gW, node, children)

if nargin < 3; children = find(gW.cp==node); end;

for c = 1:numel(children)
    
    trainData = gW.Projections{children(c)} - repmat(gW.Centers{children(c)}, 1,gW.Sizes(children(c))); 
    trainData = trainData - gW.ScalFuns{node}*(gW.ScalFuns{node}'*trainData);
    
    initdict = gW.WavBases{children(c)};
    if ~gW.isaleaf(children(c)), 
        initdict = [initdict gW.ScalFuns{node}];  %#ok<AGROW>
        %initdict = [initdict gW.X(randsample(gW.Sizes(node), size(initdict,2)), :)']; %#ok<AGROW>
    end
    
    if ~isempty(initdict) %&& gW.Sizes(children(c)) > size(initdict,2)
        initdict2 = [initdict, trainData(:,1:min([(gW.opts.sparsifying_oversampling-1)*size(initdict,2),size(trainData,2)-size(initdict,2)]))];
        switch lower(gW.opts.sparsifying_method)            
            case 'ksvd'
                D  = ksvd(struct('data', trainData, 'Edata', gW.opts.threshold0(node)*max(sum(trainData.^2,1)), 'iternum', 30, 'initdict', initdict2,'muthresh',0.9),'rt');
                D = full(D);
                lsvd = svd(D);
                lNumberOfAtoms = length(find(lsvd/lsvd(1)>0.1));
                if lNumberOfAtoms<size(D,2),
                    fprintf('Compressing out %d sparse atoms.\n',size(D,2)-lNumberOfAtoms);
                    D  = ksvd(struct('data', trainData, 'Edata', gW.opts.threshold0(node)*max(sqrt(sum(trainData.^2,1))), 'iternum', 30, 'initdict', D(:,1:lNumberOfAtoms),'muthresh',0.9),'rt');
                    D = full(D);
                end;
                %DisplayImageCollection(reshape([D],16,16,[]),50,1,1);title('K-SVD dictionary');DisplayImageCollection(reshape([initdict],16,16,[]),50,1,1);title('initial dictionary');DisplayImageCollection(reshape([trainData],16,16,[]),50,1,1);title('Local data');
            case 'spams'
                param.lambda= 1e-6;
                param.iter = -5; % let's wait only 10 seconds
                param.mode = 1;
                param.modeD = 0;
                param.D = initdict;
                %param.K = min(gW.Sizes(children(c)), size(gW.WavBases{children(c)},2));
                D = mexTrainDL(trainData',param);
        end
        gW.WavBases{children(c)} = D;
    end

end

% %sparsify scaling functions
% if gW.opts.addTangentialCorrections
%     trainData = gW.X(gW.PointsInNet{node}, :) - cat(1, gW.Projections{children});
%     trainData = trainData - (trainData*gW.ScalFuns{node})*gW.ScalFuns{node}';
%     gW.ScalFuns{node} = ksvd(struct('data', trainData', 'Edata', 1e-5, 'iternum', 10, 'dictsize', 2*size(gW.ScalFuns{node},2)));
% end

return;
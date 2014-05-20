function [gW, Data] = simplify_the_GWT_tree(gW, Data)

J = max(gW.Scales);
a(gW.LeafNodes) = 1:numel(gW.LeafNodes);

flags = ones(1,length(gW.cp));

for j = J:-1:1
    
    nodes = find(gW.Scales == j);
    
    for i = 1:numel(nodes)
        
        node = nodes(i);
        
        [~, leafnodeOffspring] = get_offspring(gW.cp, node);
        
%         % compress scaling functions
%         if ~gW.isaleaf(node)
%             
%             matCoeffs = cat(1,Data.CelTangCoeffs{a(leafnodeOffspring),j+1});
%             
%             nzCols = find(mean(abs(matCoeffs), 1)>1e-10);
%             nnzCols = numel(nzCols);
%             
%             if nnzCols < size(matCoeffs,2)
%                 
%                 gW.ScalFuns{node} = gW.ScalFuns{node}(:, nzCols);
%                 for n = 1:length(leafnodeOffspring)
%                     Data.CelTangCoeffs{a(leafnodeOffspring(n)),j+1} = Data.CelTangCoeffs{a(leafnodeOffspring(n)),j+1}(:, nzCols);
%                 end              
%                 
%             end
%             
%         end
        
        % compress wavelet bases and coefficients
        matCoeffs = cat(1,Data.CelWavCoeffs{a(leafnodeOffspring),j});
        
        if ~isempty(matCoeffs)
            nzCols = find(mean(matCoeffs.^2,1)>1e-12);
            nnzCols = numel(nzCols);
        else 
            nnzCols = 0;
        end
        
        if isempty(matCoeffs) || nnzCols < size(matCoeffs,2)
            
            if nnzCols>0 % shrink both basis and coefficient
                
                gW.WavBases{node} = gW.WavBases{node}(:, nzCols);
                for n = 1:length(leafnodeOffspring)
                    Data.CelWavCoeffs{a(leafnodeOffspring(n)),j} = Data.CelWavCoeffs{a(leafnodeOffspring(n)),j}(:, nzCols);
                end
                
%                 if gW.opts.pruning,
%                     gW.epsEncodingCosts(node) = gW.epsEncodingCosts(node) - ...
%                         (gW.Sizes(node)+gW.opts.AmbientDimension)*(size(gW.WavBases{node},2)-nnzCols);
%                 end
                
            else
                
                gW.WavBases{node} = [];
                for n = 1:length(leafnodeOffspring)
                    %Data.CelWavCoeffs{a(leafnodeOffspring(n)),j} = [];
                    Data.CelWavCoeffs{a(leafnodeOffspring(n)),j} = Data.CelWavCoeffs{a(leafnodeOffspring(n)),j}(:,1:0);
                end
                
                if ~gW.isaleaf(node)
                    
                    children = find(gW.cp==node);
                                    
                    gW.cp(children) = gW.cp(node);
                    gW.cp(node) = -1;
                    flags(node) = -1;
                    
                    for c = 1:length(children)
                        gW.WavConsts{children(c)} = gW.WavConsts{children(c)}+gW.WavConsts{node};
                    end
                    
                    Data.CelWavCoeffs(a(leafnodeOffspring), j:J-1) = Data.CelWavCoeffs(a(leafnodeOffspring), j+1:J);
                    %Data.CelWavCoeffs(a(leafnodeOffspring), J) = [];
                   
                end
                
            end
        
        end
        
    end
        
end

%%
% leafnodesParents = gW.cp(gW.LeafNodes);
% v = zeros(1, length(gW.cp));
% v(leafnodesParents) = 1;
% leafnodesParents = find(v>0);
% 
% for i = 1:length(leafnodesParents)
%     node = leafnodesParents(i);
%     children = find(gW.cp==node);
%     if all(gW.isaleaf(children)) && isempty([gW.WavBases{children}]) && max(sum(cat(1,gW.WavConsts{children}).^2, 2))<1e-6,
%         gW.cp(children)=-1;
%         flags(children)=-1;
%         gW.isaleaf(node) = 1;
%     end
% end

%% 
gW = get_subtree(gW, flags);
gW.DictCosts = computeDictionaryCost(gW);
%%
Data.CelWavCoeffs = Data.CelWavCoeffs(:,1:max(gW.Scales));

[Data.MatWavCoeffs, Data.maxWavDims, Data.MatWavDims] = cell2mat_coeffs(Data.CelWavCoeffs, cumsum(gW.Sizes(gW.LeafNodes)));
% [Data.MatScalCoeffs, Data.maxScalDims, Data.MatScalDims] = cell2mat_coeffs(Data.CelTangCoeffs);

Data.CoeffsCosts = sum(sum(abs(Data.MatWavCoeffs)>0));


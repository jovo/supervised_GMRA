function [GMRA_Classifier, GMRA_train] = GMRA_LDA( X, Labels, Opts )

%
% function [GMRA_Classifier, GMRAs] = GMRA_LDA( X, Labels, Opts )
%
% IN:
%   X       : D by N matrix of N points in D dimensions
%   Labels  : N vector of labels for the points
%   [Opts]  : structure with options
%               [GMRAopts] : options for the GMRA. Default: uses GMRA defaults.
%               [CVsplits] : how many CV splits to do. Default: 5.
%
%
% OUT:
%   GMRA_Classifier : a classifier associated with the GMRA
%   GMRA_train      : cell array of GMRA constructed during training phase
%

if nargin<3,    Opts = []; end;
if ~isfield(Opts,'GMRAopts'), Opts.GMRAopts = []; end;
if ~isfield(Opts,'CVsplits'), Opts.CVsplits = 5; end;

GMRA_Classifier = []; 
GMRA_train = [];

[D,N] = size(X);

%% Hold out points

% Number of sets to break data into for holdout
% Can't continue if there aren't as many data points as holdout groups
if N < Opts.CVsplits,
    error('\n GMRA_LDA: number of holdout groups larger than number of points');;
end
    
% This method assures that we have at least one point per group
random_indices = randperm(N);
% Mod 3 of original indices would give us [0 1 2 0 1 2 ...]
zero_based_groups = mod(random_indices, Opts.CVsplits);
% But group labels are 1-based
groups = zero_based_groups + 1;

% This parameter sets the depth down in the GMRA tree that will be searched
% past the point at which using the children is worse than using the node
% itself. DEPTH = 0 will stop immediately when children don't help. 
% DEPTH = 2 will look down 2 levels to see if it can do better. I usually
% set to 6 or 10 to search most of the tree.
ALLOWED_DEPTH = 2;

% Flag for error status on each node
USE_THIS = 10;
USE_SELF = 1;
USE_CHILDREN = -1;
UNDECIDED = -10;

LDA_DIM_LIMIT = 100;

results_cell = cell(Opts.CVsplits, 1);
results_holdout_cell = cell(Opts.CVsplits, 1);

% Outer loop over holdout groups
for rr = 1:Opts.CVsplits,
    % Pull out groups for holdout
    X_train = X(:,groups ~= rr);
    X_test = X(:,groups == rr);
    imgOpts.Labels_train = Labels(groups ~= rr);
    imgOpts.Labels_test = Labels(groups == rr);
    
    %% Do straight LDA on all the data, possibly with reduced dimensionality
    
    % Dimensionality into which to project data for straight LDA (0 = no dim reduction)
    if size(X_train,1) > LDA_DIM_LIMIT,
        straight_lda_dim = LDA_DIM_LIMIT;
        % Dim must be smaller than number of points
        if size(X_train,2) < straight_lda_dim,
            straight_lda_dim = size(X_train,2) - 1;
        end
    else
        straight_lda_dim = 0;
    end
    
    if ((straight_lda_dim > 0) && (straight_lda_dim < size(X_train,1))),
        X0 = X_train;
        cm = mean(X0,2);
        X1 = X0 - repmat(cm, 1, size(X0,2));
        clear('X0');
        fprintf('pre-LDA randPCA from %d to %d dimensions\n', size(X1,1), straight_lda_dim);
        % NOTE: randPCA calls RAND
        [~,S,V] = randPCA(X1, straight_lda_dim);
        X_lda = S*V';
        clear('X1', 'S', 'V');
    else
        X_lda = X_train;
    end;
    
    fprintf(1, 'Straight LDA in %d dim\n', straight_lda_dim);
    [straight_lda_error, straight_lda_std] = lda_multi_crossvalidation(X_lda, imgOpts.Labels_train);
    straight_lda_complexity = length(unique(imgOpts.Labels_train)) * straight_lda_dim^2;

    %% Generate GWT

    % Construct geometric wavelets
    fprintf(1, 'GMRA, Group %d / %d\n', rr, Opts.CVsplits);
    GWT = GMRA(X_train, Opts.GMRAopts);

    % Getting rid of GWT.X to make sure train/test code isn't screwed up...
    GWT = rmfield(GWT, 'X');
    GWT.X_train = X_train;
    GWT.X_test = X_test;

    % Get rid of this original since have copy in GWT.X
    %clear('X_train','X_test');

    % Compute all wavelet coefficients 
    fprintf(1, 'GWT Training Data\n\n');
    [GWT, Data_train] = GWT_trainingData(GWT, GWT.X_train);
    
    % Place held out points into existing GWT tree
    Data_test = FGWT(GWT, GWT.X_test);

    % NOTE: Deleting some unneeded data for memory's sake
    % Data_train = rmfield(Data_train, 'Projections');
    % Data_train = rmfield(Data_train, 'TangentialCorrections');
    % Data_train = rmfield(Data_train, 'MatWavCoeffs');
    % Data_test = rmfield(Data_test, 'MatWavCoeffs');


    %% Build model with train data split by cross-validation

    fprintf(1, 'LDA\n');

    n_pts_train = length(imgOpts.Labels_train);

    % Combined uses both scaling functions and wavelets together for all fine
    % scales. Otherwise, only scaling functions are used for all scales.
    COMBINED = false;

    results = struct();

    % NOTE: For now testing out initializing all results so array will be
    %  the right length later on, but init values = NaN and values tested but
    %  too big = Inf
    for ii = 1:length(GWT.cp),
        results(ii).self_error = NaN;
        results(ii).self_std = NaN;
        results(ii).best_children_errors = NaN;
        results(ii).direct_children_errors = NaN;
        results(ii).error_value_to_use = NaN;
    end

    % Version that tests holdout by walking tree only down as far as it needs
    % before it hits single class or too small nodes

    tree_parent_idxs = GWT.cp;

    % Container for child nodes which need to be freed up if a node eventually
    % switches from USE_SELF to USE_CHILDREN, but some children have stopped
    % propagating down the tree because they were past the ALLOWED_DEPTH of
    % the indexed node
    children_to_free = cell([length(GWT.cp) 1]);

    % Start at root of the tree (cp(root_idx) == 0)
    root_idx = find(tree_parent_idxs == 0);

    if (length(root_idx) > 1)
        fprintf( 'cp contains too many root nodes (cp == 0)!! \n' );
        return;
    else
        % This routine calculates errors for the children of the current node
        % so we need to first calculate the root node error
        [total_errors, std_errors] = gwt_single_node_lda_crossvalidation( GWT, Data_train, imgOpts.Labels_train, root_idx, COMBINED );

        % Record the results for the root node
        results(root_idx).self_error = total_errors;
        results(root_idx).self_std = std_errors;
        results(root_idx).error_value_to_use = UNDECIDED;
        % fprintf( 'current node: %d\n', root_idx );

        % The java deque works with First = most recent, Last = oldest
        %   so since it can be accessed with removeFirst / removeLast
        %   it can be used either as a LIFO stack or FIFO queue
        % Here I'm trying it as a deque/queue to do a breadth-first tree
        %   traversal
        node_idxs = java.util.ArrayDeque();
        node_idxs.addFirst(root_idx);

        % Main loop to work iteratively down the tree breadth first
        while (~node_idxs.isEmpty())
            current_node_idx = node_idxs.removeLast();
            % fprintf( 'current node: %d\n', current_node_idx );

            % Get list of parent node indexes for use in a couple spots later
            % TODO: Move to a routine...
            current_parents_idxs = [];
            tmp_current_idx = current_node_idx;
            while (tree_parent_idxs(tmp_current_idx) > 0), 
                tmp_parent_idx = tree_parent_idxs(tmp_current_idx); 
                current_parents_idxs(end+1) = tmp_parent_idx; 
                tmp_current_idx = tmp_parent_idx; 
            end

            % Get children of the current node
            current_children_idxs = find(tree_parent_idxs == current_node_idx);

            % Loop through the children
            for current_child_idx = current_children_idxs,

                % Calculate the error on the current child
                [total_errors, std_errors] = gwt_single_node_lda_crossvalidation( GWT, Data_train, imgOpts.Labels_train, current_child_idx, COMBINED );

                % Record the results for the current child
                results(current_child_idx).self_error = total_errors;
                results(current_child_idx).self_std = std_errors;
                results(current_child_idx).error_value_to_use = UNDECIDED;
                % fprintf( '\tchild node: %d\n', current_child_idx );
            end

            % If no children, want error to be infinite for any comparisons
            children_error_sum = Inf;
            % Set children errors to child sum (if there are children because sum([]) == 0)
            if ~isempty(current_children_idxs)
                children_error_sum = sum( [results(current_children_idxs).self_error] );
                results(current_node_idx).direct_children_errors = children_error_sum;
                results(current_node_idx).best_children_errors = children_error_sum;
            end

            % Compare children results to self error
            self_error = results(current_node_idx).self_error;
            % NOTE: Here is where to put some slop based on standard deviation
            if (self_error < children_error_sum)
                % Set status = USE_SELF
                results(current_node_idx).error_value_to_use = USE_SELF;

            else
                % Set status = USE_CHILDREN
                results(current_node_idx).error_value_to_use = USE_CHILDREN;

                % Propagate difference up parent chain
                error_difference = self_error - children_error_sum;
                % DEBUG
                % fprintf('Node %d has %d error difference\n', current_node_idx, error_difference);
                % Loop through list of parent nodes
                for parent_node_idx = current_parents_idxs,

                    % Subtract difference from best_children_errors
                    % DEBUG
                    % fprintf('\tParent node %d best children error %d\n', parent_node_idx, results(parent_node_idx).best_children_errors);
                    results(parent_node_idx).best_children_errors = results(parent_node_idx).best_children_errors - error_difference;
                    % DEBUG
                    % fprintf('\t\tnow down to %d\n', results(parent_node_idx).best_children_errors);

                    % If parent.status = USE_CHILDREN
                    if (results(parent_node_idx).error_value_to_use == USE_CHILDREN)
                        % Propagate differnce up to parent
                        continue;

                    % else if parent.status = USE_SELF
                    elseif (results(parent_node_idx).error_value_to_use == USE_SELF)
                        % Compare best_children_errors to self_error
                        % NOTE: Here again use same slop test as above...

                        % if still parent.self_error < parent.best_children_errors
                        if (results(parent_node_idx).self_error < results(parent_node_idx).best_children_errors),
                            % stop difference propagation
                            break;
                        % else if now parent.best_children_errors < parent.self_error
                        else
                            % parent.status = USE_CHILDREN
                            results(parent_node_idx).error_value_to_use = USE_CHILDREN;
                            % propagate this NEW difference up to parent
                            error_difference = results(parent_node_idx).self_error - results(parent_node_idx).best_children_errors;
                            % Since some children of this node might have
                            % not added their children to the queue because
                            % this node was too far up the tree for
                            % ALLOWED_DEPTH, now that this has switched, need
                            % to check those older nodes to see if now their
                            % children should be added...
                            for idx = children_to_free{parent_node_idx}
                                node_idxs.addFirst(idx);
                                % DEBUG
                                % fprintf(' * *   freeing: %d\n', idx);
                            end
                            children_to_free{parent_node_idx} = [];
                           continue;
                        end
                    else
                        fprintf('\nERROR: parent error status flag not set properly on index %d!!\n', parent_node_idx);
                    end
                end
            end

            % Allowing here to go a certain controlled depth beyond where
            %   the children seem to be worse than a parent to see if it
            %   eventually reverses. Set threshold to Inf to use whole tree
            %   of valid error values. Set threshold to zero to never go beyond
            %   a single reversal where children are greater than the parent.

            % Figure out how far up tree to highest USE_SELF
            % If hole_depth < hole_depth_threshold
                % Push children on to queue for further processing
            % else
                % stop going any deeper
            self_parent_idx_chain = [current_node_idx current_parents_idxs];
            self_parent_status_flags = [results(self_parent_idx_chain).error_value_to_use];
            use_self_depth = find(self_parent_status_flags == USE_SELF, 1, 'last');
            % Depth set with this test
            % Root node or not found gives empty find result

            use_self_depth_low_enough = isempty(use_self_depth) || (use_self_depth <= ALLOWED_DEPTH);

            % All children must have finite error sums to go lower in any child
            all_children_errors_finite = isfinite(children_error_sum);

            % If child errors are finite, but the node is too deep, keep track
            % of the child indexes to add to the queue in case the USE_SELF
            % node it's under switches to USE_CHILDREN
            if (~use_self_depth_low_enough && all_children_errors_finite)
                problem_parent_idx = self_parent_idx_chain(use_self_depth);
                children_to_free{problem_parent_idx} = cat(2, children_to_free{problem_parent_idx}, current_children_idxs);
            end

            % Only addFirst children on to the stack if this node qualifies
            if (use_self_depth_low_enough && all_children_errors_finite)
                % Find childrent of current node
                % DEBUG
                % fprintf(' + + Current index: %d\n', current_node_idx);
                for idx = current_children_idxs
                    node_idxs.addFirst(idx);
                    % DEBUG
                    % fprintf(' + +   adding: %d\n', idx);
                end
            else
                % DEBUG
                % fprintf(' - - Current index: %d\n', current_node_idx);
                % fprintf(' - -   use_self_depth: %d, all_children_errors_finite: %d\n', use_self_depth_low_enough, all_children_errors_finite);
            end
        end
    end

    %% Now go through tree using training model
    %   but use test data split and find best model (nodes) for that data

    results_holdout = struct();

    % NOTE: For now testing out initializing all results so array will be
    %  the right length later on, but init values = NaN and values tested but
    %  too big = Inf
    for ii = 1:length(GWT.cp),
        results_holdout(ii).self_error = NaN;
        results_holdout(ii).self_std = NaN;
        results_holdout(ii).best_children_errors = NaN;
        results_holdout(ii).direct_children_errors = NaN;
        results_holdout(ii).error_value_to_use = NaN;
    end

    % Container for child nodes which need to be freed up if a node eventually
    % switches from USE_SELF to USE_CHILDREN, but some children have stopped
    % propagating down the tree because they were past the ALLOWED_DEPTH of
    % the indexed node
    children_to_free = cell([length(GWT.cp) 1]);

    if (length(root_idx) > 1)
        fprintf( 'cp contains too many root nodes (cp == 0)!! \n' );
        return;
    else
        % This routine calculates errors for the children of the current node
        % so we need to first calculate the root node error
        [total_errors, model_train] = gwt_single_node_lda_traintest( GWT, Data_train, Data_test, imgOpts, root_idx, COMBINED );

        % Record the results for the root node
        results_holdout(root_idx).self_error = total_errors;
        results_holdout(root_idx).self_std = std_errors;
        results_holdout(root_idx).error_value_to_use = UNDECIDED;
        % fprintf( 'current node: %d\n', root_idx );

        % The java deque works with First = most recent, Last = oldest
        %   so since it can be accessed with removeFirst / removeLast
        %   it can be used either as a LIFO stack or FIFO queue
        % Here I'm trying it as a deque/queue to do a breadth-first tree
        %   traversal
        node_idxs = java.util.ArrayDeque();
        node_idxs.addFirst(root_idx);

        % Main loop to work iteratively down the tree breadth first
        while (~node_idxs.isEmpty())
            current_node_idx = node_idxs.removeLast();
            % fprintf( 'current node: %d\n', current_node_idx );

            % Get list of parent node indexes for use in a couple spots later
            % TODO: Move to a routine...
            current_parents_idxs = [];
            tmp_current_idx = current_node_idx;
            while (tree_parent_idxs(tmp_current_idx) > 0), 
                tmp_parent_idx = tree_parent_idxs(tmp_current_idx); 
                current_parents_idxs(end+1) = tmp_parent_idx; 
                tmp_current_idx = tmp_parent_idx; 
            end

            % Get children of the current node
            current_children_idxs = find(tree_parent_idxs == current_node_idx);

            % Loop through the children
            for current_child_idx = current_children_idxs,

                % Calculate the error on the current child
                [total_errors, model_train] = gwt_single_node_lda_traintest( GWT, Data_train, Data_test, imgOpts, current_child_idx, COMBINED );

                % Record the results for the current child
                results_holdout(current_child_idx).self_error = total_errors;
                results_holdout(current_child_idx).self_std = std_errors;
                results_holdout(current_child_idx).error_value_to_use = UNDECIDED;
                % fprintf( '\tchild node: %d\n', current_child_idx );
            end

            % If no children, want error to be infinite for any comparisons
            children_error_sum = Inf;
            % Set children errors to child sum (if there are children because sum([]) == 0)
            if ~isempty(current_children_idxs)
                children_error_sum = sum( [results_holdout(current_children_idxs).self_error] );
                results_holdout(current_node_idx).direct_children_errors = children_error_sum;
                results_holdout(current_node_idx).best_children_errors = children_error_sum;
            end

            % Compare children results to self error
            self_error = results_holdout(current_node_idx).self_error;
            % NOTE: Here is where to put some slop based on standard deviation
            if (self_error < children_error_sum)
                % Set status = USE_SELF
                results_holdout(current_node_idx).error_value_to_use = USE_SELF;

            else
                % Set status = USE_CHILDREN
                results_holdout(current_node_idx).error_value_to_use = USE_CHILDREN;

                % Propagate difference up parent chain
                error_difference = self_error - children_error_sum;
                % DEBUG
                % fprintf('Node %d has %d error difference\n', current_node_idx, error_difference);
                % Loop through list of parent nodes
                for parent_node_idx = current_parents_idxs,

                    % Subtract difference from best_children_errors
                    % DEBUG
                    % fprintf('\tParent node %d best children error %d\n', parent_node_idx, results_holdout(parent_node_idx).best_children_errors);
                    results_holdout(parent_node_idx).best_children_errors = results_holdout(parent_node_idx).best_children_errors - error_difference;
                    % DEBUG
                    % fprintf('\t\tnow down to %d\n', results_holdout(parent_node_idx).best_children_errors);

                    % If parent.status = USE_CHILDREN
                    if (results_holdout(parent_node_idx).error_value_to_use == USE_CHILDREN)
                        % Propagate differnce up to parent
                        continue;

                    % else if parent.status = USE_SELF
                    elseif (results_holdout(parent_node_idx).error_value_to_use == USE_SELF)
                        % Compare best_children_errors to self_error
                        % NOTE: Here again use same slop test as above...

                        % if still parent.self_error < parent.best_children_errors
                        if (results_holdout(parent_node_idx).self_error < results_holdout(parent_node_idx).best_children_errors),
                            % stop difference propagation
                            break;
                        % else if now parent.best_children_errors < parent.self_error
                        else
                            % parent.status = USE_CHILDREN
                            results_holdout(parent_node_idx).error_value_to_use = USE_CHILDREN;
                            % propagate this NEW difference up to parent
                            error_difference = results_holdout(parent_node_idx).self_error - results_holdout(parent_node_idx).best_children_errors;
                            % Since some children of this node might have
                            % not added their children to the queue because
                            % this node was too far up the tree for
                            % ALLOWED_DEPTH, now that this has switched, need
                            % to check those older nodes to see if now their
                            % children should be added...
                            for idx = children_to_free{parent_node_idx}
                                node_idxs.addFirst(idx);
                                % DEBUG
                                % fprintf(' * *   freeing: %d\n', idx);
                            end
                            children_to_free{parent_node_idx} = [];
                           continue;
                        end
                    else
                        fprintf('\nERROR: parent error status flag not set properly on index %d!!\n', parent_node_idx);
                    end
                end
            end

            % Allowing here to go a certain controlled depth beyond where
            %   the children seem to be worse than a parent to see if it
            %   eventually reverses. Set threshold to Inf to use whole tree
            %   of valid error values. Set threshold to zero to never go beyond
            %   a single reversal where children are greater than the parent.

            % Figure out how far up tree to highest USE_SELF
            % If hole_depth < hole_depth_threshold
                % Push children on to queue for further processing
            % else
                % stop going any deeper
            self_parent_idx_chain = [current_node_idx current_parents_idxs];
            self_parent_status_flags = [results_holdout(self_parent_idx_chain).error_value_to_use];
            use_self_depth = find(self_parent_status_flags == USE_SELF, 1, 'last');
            % Depth set with this test
            % Root node or not found gives empty find result

            use_self_depth_low_enough = isempty(use_self_depth) || (use_self_depth <= ALLOWED_DEPTH);

            % All children must have finite error sums to go lower in any child
            all_children_errors_finite = isfinite(children_error_sum);

            % If child errors are finite, but the node is too deep, keep track
            % of the child indexes to add to the queue in case the USE_SELF
            % node it's under switches to USE_CHILDREN
            if (~use_self_depth_low_enough && all_children_errors_finite)
                problem_parent_idx = self_parent_idx_chain(use_self_depth);
                children_to_free{problem_parent_idx} = cat(2, children_to_free{problem_parent_idx}, current_children_idxs);
            end

            % Only addFirst children on to the stack if this node qualifies
            if (use_self_depth_low_enough && all_children_errors_finite)
                % Find childrent of current node
                % DEBUG
                % fprintf(' + + Current index: %d\n', current_node_idx);
                for idx = current_children_idxs
                    node_idxs.addFirst(idx);
                    % DEBUG
                    % fprintf(' + +   adding: %d\n', idx);
                end
            else
                % DEBUG
                % fprintf(' - - Current index: %d\n', current_node_idx);
                % fprintf(' - -   use_self_depth: %d, all_children_errors_finite: %d\n', use_self_depth_low_enough, all_children_errors_finite);
            end
        end
    end
    
    %% Only evaluate the held-out points on the training crossvalidation winner model (nodes)
    
    % Traverse the tree and mark the winner nodes from the cross-validation
    
    % The java deque works with First = most recent, Last = oldest
    %   so since it can be accessed with removeFirst / removeLast
    %   it can be used either as a LIFO stack or FIFO queue
    % Here I'm trying it as a stack to do a depth-first tree traversal
    node_idxs = java.util.ArrayDeque();
    node_idxs.addFirst(root_idx);
    
    % Keep track of the total holdout errors at "optimal" scales
    total_optimal_holdout_data_error = 0;
    % Keep track of the sum of node "complexities" (#cats * dim(node)^2)
    total_optimal_complexity = 0;

    % Main loop to work iteratively down the tree depth first
    while (~node_idxs.isEmpty())
        current_node_idx = node_idxs.removeFirst();
        % DEBUG
        % fprintf( 'current node: %d\n', current_node_idx );
        
        % Set flag if this is the deepest node to use
        if (results(current_node_idx).error_value_to_use == USE_SELF)
            results(current_node_idx).error_value_to_use = USE_THIS;
            holdout_error = gwt_single_node_lda_traintest( GWT, Data_train, Data_test, imgOpts, current_node_idx, COMBINED );
            node_complexity = gwt_single_node_complexity( GWT, Data_train, imgOpts, current_node_idx, COMBINED );
            total_optimal_holdout_data_error = total_optimal_holdout_data_error + holdout_error;
            total_optimal_complexity = total_optimal_complexity + node_complexity;
        else
            % Get children of the current node
            current_children_idxs = find(tree_parent_idxs == current_node_idx);

            % and put them in the stack for further traversal
            for idx = current_children_idxs
                node_idxs.addFirst(idx);
                % DEBUG
                % fprintf(' + +   adding: %d\n', idx);
            end
        end
    end

    %% For visualization, mark optimal hold-out test data winner nodes
    %   but at the same time sum up training error on this optimal test
    %   model
    
    % Traverse the tree and mark the winner nodes from the cross-validation
    
    % The java deque works with First = most recent, Last = oldest
    %   so since it can be accessed with removeFirst / removeLast
    %   it can be used either as a LIFO stack or FIFO queue
    % Here I'm trying it as a stack to do a depth-first tree traversal
    node_idxs = java.util.ArrayDeque();
    node_idxs.addFirst(root_idx);

    % Keep track of the total training data errors at "optimal" holdout data scales
    total_holdout_model_train_data_err = 0;

    % Main loop to work iteratively down the tree depth first
    while (~node_idxs.isEmpty())
        current_node_idx = node_idxs.removeFirst();
        % DEBUG
        % fprintf( 'current node: %d\n', current_node_idx );
        
        % Set flag if this is the deepest node to use
        if (results_holdout(current_node_idx).error_value_to_use == USE_SELF)
            results_holdout(current_node_idx).error_value_to_use = USE_THIS;
            total_holdout_model_train_data_err = total_holdout_model_train_data_err + results(current_node_idx).self_error;
        else
            % Get children of the current node
            current_children_idxs = find(tree_parent_idxs == current_node_idx);

            % and put them in the stack for further traversal
            for idx = current_children_idxs
                node_idxs.addFirst(idx);
                % DEBUG
                % fprintf(' + +   adding: %d\n', idx);
            end
        end
    end

    
    %% Tree of results
    % http://stackoverflow.com/questions/5065051/add-node-numbers-get-node-locations-from-matlabs-treeplot

    H = figure;
    treeplot(GWT.cp, 'g.', 'c');

    % treeplot is limited with control of colors, etc.
    P = findobj(H, 'Color', 'c');
    set(P, 'Color', [247 201 126]/255);
    P2 = findobj(H, 'Color', 'g');
    set(P2, 'MarkerSize', 5, 'Color', [180 180 180]/255);

    % count = size(GWT.cp,2);
    [x,y] = treelayout(GWT.cp);
    x = x';
    y = y';
    hold();

    % Show which nodes were used (self or children)
    ee = [results(:).error_value_to_use];
    use_self_bool = ee == USE_THIS;
    plot(x(use_self_bool), y(use_self_bool), '.', 'MarkerSize', 50, 'Color', [0.8 0.5 0.5]);

    ee_h = [results_holdout(:).error_value_to_use];
    h_use_self_bool = ee_h == USE_THIS;
    plot(x(h_use_self_bool), y(h_use_self_bool), 'o', 'MarkerSize', 10, 'Color', [0.8 0.5 0.5]);

    error_array = round([results(:).self_error]);
    error_strings = cellstr(num2str(error_array'));
    std_array = round([results(:).self_std]);
    std_strings = cellstr(num2str(std_array'));
    % nptsinnode_strings = cellstr(num2str((cellfun(@(x) size(x,2), GWT.PointsInNet))'));
    cp_idx_strings = cellstr(num2str((1:length(GWT.cp))'));

    childerr_strings = cell(length(GWT.cp),1);
    for ii = 1:length(GWT.cp),
        childerr_strings{ii} = num2str(round(results(ii).direct_children_errors));
    end

    % combo_strings = strcat(error_strings, '~', std_strings);
    % childcombo_strings = strcat(childerr_strings, '~', childstd_strings);
    combo_strings = error_strings;
    childcombo_strings = childerr_strings;

    besterr_strings = cell(length(GWT.cp),1);
    for ii = 1:length(GWT.cp),
        besterr_strings{ii} = num2str(round(results(ii).best_children_errors));
    end

    holderr_strings = cell(length(GWT.cp),1);
    for ii = 1:length(GWT.cp),
        holderr_strings{ii} = num2str(round(results_holdout(ii).self_error));
    end

    childholderr_strings = cell(length(GWT.cp),1);
    for ii = 1:length(GWT.cp),
        childholderr_strings{ii} = num2str(round(results_holdout(ii).direct_children_errors));
    end

    bestholderr_strings = cell(length(GWT.cp),1);
    for ii = 1:length(GWT.cp),
        bestholderr_strings{ii} = num2str(round(results_holdout(ii).best_children_errors));
    end

    % Only displaying the finite values for now
    finite_errors = isfinite(str2double(combo_strings));
    finite_childerr = isfinite(str2double(childcombo_strings));
    finite_besterr = isfinite(str2double(besterr_strings));

    % Node errors
    text(x(finite_errors), y(finite_errors)+0.01, combo_strings(finite_errors), ...
        'VerticalAlignment','bottom','HorizontalAlignment','right','Color', [0.2 0.2 0.2]);
    % Child node errors
    text(x(finite_childerr), y(finite_childerr), childcombo_strings(finite_childerr), ...
        'VerticalAlignment','middle','HorizontalAlignment','right','Color', [0.4 0.2 0.2])
    % Best Child node errors
    text(x(finite_besterr), y(finite_besterr)-0.01, besterr_strings(finite_besterr), ...
        'VerticalAlignment','top','HorizontalAlignment','right','Color', [0.2 0.4 0.2]);

    % Total training data error at "optimal" holdout data scales + orig
    % model "complexity"
    text(x(end), y(end)-0.04, [num2str(round(total_holdout_model_train_data_err)) '·' num2str(total_optimal_complexity)], ...
        'VerticalAlignment','top','HorizontalAlignment','right','Color', [0.2 0.2 0.4]);
    
    % Straight LDA error on training data
    text(x(end), y(end)+0.04, [num2str(round(straight_lda_error)) '·' num2str(straight_lda_dim) '·' num2str(straight_lda_complexity)], ...
        'VerticalAlignment','bottom','HorizontalAlignment','right','Color', [0.2 0.2 0.4]);

    % Node cp index
    % text(x(:,1), y(:,1), cp_idx_strings, ...
    %     'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[0.2 0.2 0.6])
    
    finite_holderr = isfinite(str2double(holderr_strings));
    finite_childholderr = isfinite(str2double(childholderr_strings));
    finite_bestholderr = isfinite(str2double(bestholderr_strings));

    % Holdout error
    text(x(finite_holderr), y(finite_holderr)+0.01, holderr_strings(finite_holderr), ...
        'VerticalAlignment','bottom','HorizontalAlignment','left','Color', [0.2 0.2 0.2]);
    % Child holdout node errors
    text(x(finite_childholderr), y(finite_childholderr), childholderr_strings(finite_childholderr), ...
        'VerticalAlignment','middle','HorizontalAlignment','left','Color', [0.4 0.2 0.2])
    % Best Holdout child error
    text(x(finite_bestholderr), y(finite_bestholderr)-0.01, bestholderr_strings(finite_bestholderr), ...
        'VerticalAlignment','top','HorizontalAlignment','left','Color', [0.2 0.4 0.2]);

    title({['LDACV: ' num2str(ALLOWED_DEPTH) ' deep ' strrep('DataSet','_', ' ') ' - ' num2str(n_pts_train) ' / ' num2str(length(imgOpts.Labels_test)) ' pts']}, ...
        'Position', [0.01 1.02], 'HorizontalAlignment', 'Left', 'Margin', 10);
    
    % Total holdout error at "optimal" scales
    text(x(end), y(end)-0.04, num2str(total_optimal_holdout_data_error), ...
        'VerticalAlignment','top','HorizontalAlignment','left','Color', [0.2 0.2 0.4]);
    
    % Refresh plots mid-loop
    drawnow;
    
    % Copy results into cell array
    results_cell{rr} = results;
    results_holdout_cell{rr} = results_holdout;
    
% End of loop over holdout groups    
end

return;

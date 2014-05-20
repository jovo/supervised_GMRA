function CelWavCoeffs = threshold_coefficients(CelWavCoeffs, opts)

if opts.coeffs_threshold == 0; return; end;

[nLeafNodes,J] = size(CelWavCoeffs);

switch lower(opts.shrinkage)
    
    case 'soft'
        for i = 1:nLeafNodes
            for j = 1:J
                if ~isempty(CelWavCoeffs{i,j})
                    CelWavCoeffs{i,j} = sign(CelWavCoeffs{i,j}).*max(0,abs(CelWavCoeffs{i,j})-opts.coeffs_threshold);
                end
            end
        end
        
    case 'hard'
        for i = 1:nLeafNodes
            for j = 1:J
                if ~isempty(CelWavCoeffs{i,j})
                    CelWavCoeffs{i,j}(abs(CelWavCoeffs{i,j})<opts.coeffs_threshold) = 0;
                end
            end
        end
        
    case 'scale'
        j_max = opts.coeffs_threshold; 
        CelWavCoeffs(:, j_max+1:J) = cell(nLeafNodes, J-j_max);

    otherwise
        error('Shrinkage method cannot be recognized!')
        
end

return
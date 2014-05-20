function Startup_GeometricWavelets(BaseDirectory)

% Startup_DiffusionGeometry adds the various directories used by the diffusion geometry code
% to the current path.  Startup_DiffusionGeometry will assume the current directory is the
% base directory unless another directory is specified


fprintf('Startup_GeometricWavelets.m: setting diffusion geometry paths ... \n');

if nargin==0
    Prefix  = [pwd filesep];
else
    Prefix  = [BaseDirectory filesep];
end;

%appendpath(([Prefix 'Diffusions']));

fprintf('Startup_GeometricWavelets.m: disabling case sensitivity warning ... \n');
warning('off','MATLAB:dispatcher:InexactMatch');

function appendpath(string)

fprintf('\t%s\\ \n', string);
addpath(genpath(string));

return;


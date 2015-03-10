function [cX]=Generate_Swissroll_BB(NofPts, Dim, Opts)

global dx

%
% function cX=Generate_Swissroll(NofPts, Dim, Opts)
%
% Generate_Swissroll generates a Swissroll manifold.
% 
% IN:
%    NofPts     : the number of points in the manifold generated
%    [Dim]      : the dimension of the manifold, if Dim=1, a curve, if Dim=2, a surface. default = 2
%    [Opts]     : structure containing the following fields:
%                   [PtsType] : 'mesh': a meshgrid type 
%                               'rand': a random uniform sampling
%                               default = rand
%
% OUT:
%     cX: NofPtsxDim array, if PtsType is mesh, not exactly. 
%
% Example: X = Generate_Swissroll(1000, 2, struct('PtsType', 'mesh'));
%          X = Generate_Swissroll(1000);
%
% SC:
%    YM: 8/19/2008
%


% Setup parameters
if nargin < 2
    Dim = 2;
end

if nargin < 3
   Opts=[];
end

if Dim > 2
    fprintf('Dim > 2, Dim is modified to 2.\n'); 
    Dim = 2;
end

if ~isfield(Opts, 'PtsType')
    Opts.PtsType = 'rand';
end

if strcmpi(Opts.PtsType,'mesh') && Dim ==2
    fprintf('The mesh type case, it may not return the exact number of points.\n');
end


% Generate data
if Dim == 1    % curve case
    if strcmpi(Opts.PtsType, 'mesh') % mesh type sampling
        dx=(3*pi/2)*(1+2*linspace(0, 1, NofPts))';
    else    % random sampling from uniform dist
        dx=(3*pi/2)*(1+2*rand(NofPts, 1));      

    end
    X1=dx.*cos(dx)/(9*pi/2);
    X2=dx.*sin(dx)/(9*pi/2);
    cX=[X1, X2];   
%     size(cX, 1)
%     figure; plot(cX(:, 1), cX(:, 2), '.');axis equal

elseif Dim == 2 % surface case
    if strcmpi(Opts.PtsType, 'mesh') % meshtype sampling
        GridN=round(sqrt(NofPts/9)); 
        dx=(3*pi/2)*(1+2*linspace(0, 1, 9*GridN));
        dz=linspace(0,1, GridN);
        dxdy=[dx.*cos(dx)/(9*pi/2); dx.*sin(dx)/(9*pi/2)];
        [meshdx, meshdz]=meshgrid(dxdy(1, :), dz);
        [meshdy, meshdz]=meshgrid(dxdy(2, :), dz);        
        meshdx=reshape(meshdx, 1, 9*GridN^2);
        meshdy=reshape(meshdy, 1, 9*GridN^2);
        meshdz=reshape(meshdz, 1, 9*GridN^2);
        cX=[meshdx;meshdz;meshdy]';        
    else   % random sampling from uniform dist
%         dx=(3*pi/2)*(1+2*rand(NofPts, 1));
        dx=rand(NofPts, 1);% dx(1) = 0;
        dy=(3*pi/2)*(1+2*dx);%  dy(1) = 0;
        X3=rand(NofPts, 1);   %    X3(1) = 0;
%         X1=dx.*cos(dx)/(9*pi/2);
%         X2=dx.*sin(dx)/(9*pi/2);
        X1=dy.*cos(dy)/(9*pi/2);
        X2=dy.*sin(dy)/(9*pi/2);
        cX=[X1, X3, X2];   
        T{1} = X1; T{2} = X3; T{3} = X2;
    end 
    size(cX, 1)
%     figure; plot3(cX(:, 1), cX(:, 2), cX(:, 3), 'rx');
%     figure; scatter3(T{1}, T{2}, T{3}, dx, linspace(0,1,size(X1,1)), 'filled');    
    figure; subplot(1,2,1); plot3k2(T, 'ColorData', dx);     
    title('Model of SwissRoll Manifold')
    axis equal
    
% MeanVar Model
x = 0:0.01:1;
y1 = x;
y2 = x+1;

subplot(1,2,2); plot(x,y1, x,y2)
title('Mean and Variance of the SwissRoll Model')
xlabel('\rho')
legend('\mu = \rho', '\sigma^{2} = \rho+1')
end
    
return;
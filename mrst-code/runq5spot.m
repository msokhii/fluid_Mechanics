% RUNQ5SPOT:
% This is the code from Listing 9 in Aarnes, Gimse and Lie (2007)
% Slightly modified by John Stockie (Feb. 2021)
%
% Use an IMPES (IMplicit Pressure Explicit Saturation) algorithm to
% simulate two-phase, immiscible, incompressible flow in a homogeneous  
% porous medium for the "quarter-five spot" problem, with water
% injected at the lower-left, oil extracted from the upper-right.

close all
Grid.Nx=64; Dx=1; Grid.hx = Dx/Grid.Nx; % Dimension in x−direction
Grid.Ny=64; Dy=1; Grid.hy = Dy/Grid.Ny; % Dimension in y−direction
Grid.Nz=1;  Dz=1; Grid.hz = Dz/Grid.Nz; % Dimension in z−direction
N=Grid.Nx*Grid.Ny;                      % Total number of grid blocks
Grid.V=Grid.hx*Grid.hy*Grid.hz;         % Cell volumes
Grid.K=ones(3,Grid.Nx,Grid.Ny,Grid.Nz); % Unit permeability
Grid.por=ones(Grid.Nx,Grid.Ny,Grid.Nz);% Unit porosity
Q=zeros(N,1); Q([1 N])=[1 -1];          % Production/injection

xplot = linspace(Grid.hx/2,Dx-Grid.hx/2,Grid.Nx); % for plotting only
yplot = linspace(Grid.hy/2,Dy-Grid.hy/2,Grid.Ny);

Fluid.swc=0.0; Fluid.sor=0.0;           % Irreducible saturations
Fluid.vw=1.0;  Fluid.vo=1.0;            % Equal viscosities (unrealistic)
% Fluid.vw=1.0;  Fluid.vo=10.0;         % TRY: oil is much more viscous
% Grid.K(1,:,:,:) = 5*Grid.K(1,:,:,:); % TRY: anisotropic permeability

S=zeros(N,1);                           % Initial (water) saturation: oil only!
nt = 28; dt = 0.7/nt;                   % Time steps
for it=1:nt
  t = it*dt;
  [P,V]=Pres(Grid,S,Fluid,Q);           % pressure solver
  S=Upstream(Grid,S,Fluid,V,Q,dt);      % saturation solver
  
  % plot filled contours at the midpoints of the grid cells
  figure(1)
  contourf(xplot, yplot, reshape(S,Grid.Nx,Grid.Ny), 11, 'k');
  axis square; caxis ([0 1]);           % equal axes and color
  title(['Saturation, t=',num2str(t)])
  colorbar
  drawnow;                              % force update of plot
  
  figure(2)
  contourf(xplot, yplot, reshape(P,Grid.Nx,Grid.Ny), 11, 'k');
  axis square
  title(['Pressure, t=',num2str(t)])
  colorbar
  drawnow
end

% linadv.m: Solve the linear advection equation.
%
% USAGE:  linadv( method, CFL, nx, ic )  
%   - method is one of: FTBS, FTFS, FTCS, CTCS/Leap-Frog,
%     Lax-Friedrichs/LF, Lax-Wendroff/LW, Beam-Warming/BW, Fromm.
%   - CFL is the desired CFL number
%   - nx is the number of grid points
%   - ic is the type of ICs (0, 1, 2) ... see below

% Date:   January 25, 2019
% Author: John Stockie
%         Department of Mathematics
%         Simon Fraser University
%
% Examples: 
% 
% 1. upwind scheme with CFL=0.5, 0.9, 0.99, 1.0, 1.01, 1.1, ...
% 2. centered scheme: oscillations and incorrect shock speed
% 3. downwind scheme: oscillations and stationary shock
% 4. Lax-Wendroff scheme
% 5. Leap-frog scheme
% *. In all cases, consider nx=100 and nx=1000
% *. In all cases, try also piecewise constant initial data. 

function linadv( method, CFL, nx, ic, bcOrder )

  global ictype;
  
  if nargin < 1
    disp(  'Usage:  linadv( method, CFL, nx, ic, bcOrder )' );
    disp( '        bcOrder = 0 (const), 1 (linear), 2 (quadratic) outflow extrapolation' );
    disp( 'Defaults: CFL=0.8, nx=100, ic=0, bcOrder=0' );
    return;
  end
  
  if nargin < 2, CFL = 0.8; end
  if nargin < 3, nx  = 100; end
  if nargin < 4
    ictype = 0;  % 0 = sin; 1 = step; 2 = gaussian
  else
    ictype = ic;
  end
  if nargin < 5, bcOrder = 0; end
  
  a    = 1.5;   % advection speed
  tend = 5;     % end time
  xmax = 10;    % domain length [0,xmax]
  
  dx = xmax/nx; % mesh spacing (constant)
  x  = 0 : dx : xmax;
  ntprint = 50; % for printing
  xex= 0 : xmax/1000 : xmax;

  maxdt = CFL * dx / a;
  nt    = ceil( tend / maxdt );
  dt    = tend / nt;
  sigma = a * dt / dx;

  u0   = uexact(x, 0, a);
  u    = u0;
  unew = 0*u;
  
  disp( ['Method: ', method] );
  disp( ['    a = ', num2str(a)] );
  disp( ['   dx = ', num2str(dx)] );
  disp( ['sigma = ', num2str(sigma)] );
  disp( ['bcOrder = ', num2str(bcOrder)] );
  disp( ['runTag  = ', runTag] );
  
  ntprint = min(nt, ntprint);
  dtprint = tend / ntprint;
  uall = zeros(ntprint+1,nx+1);
  uall(1,:) = u0;
  uerr0 = zeros(1,nt);
  uerr1 = zeros(1,nt);
  uerr2 = zeros(1,nt);
  
  ip = 1;
  figure(1); clf
  
  for i = 1 : nt
  
    t = i*dt;
    uex = uexact(xex, t, a);
  
    switch lower(method)
  
     case {'downwind','ftfs'}
      unew(1:end-1) = u(1:end-1) - sigma * (u(2:end) - u(1:end-1));
      unew(end) = 0.0;
  
     case 'ftcs'
      unew(2:end-1) = u(2:end-1) - 0.5*sigma * (u(3:end) - u(1:end-2));
      unew(1)   = 0.0;
      unew(end) = 0.0;
  
     case {'leap-frog','ctcs'}
      if i == 1
        % Use upwind for the first step.
        unew(2:end) = u(2:end) - sigma * (u(2:end) - u(1:end-1));
        unew(1) = 0.0;
      else
        % interior
        unew(2:end-1) = uold(2:end-1) - sigma * (u(3:end) - u(1:end-2));
        unew(1) = 0.0;
  
        uFic = ficValue(u,bcOrder); 
        unew(end) = uold(end) - sigma * (uFic - u(end-1));
      end
  
     case {'lax-friedrichs','lf'}
      unew(2:end-1) = 0.5 * (u(3:end) + u(1:end-2)) - sigma/2 * ...
          (u(3:end) - u(1:end-2));
      unew(1)   = 0.0;
      unew(end) = 0.0;
  
     case {'lax-wendroff','lw'}
      unew(2:end-1) = u(2:end-1) - 0.5*sigma * (u(3:end) - u(1:end-2)) ...
          + 0.5*sigma^2 * (u(3:end)-2*u(2:end-1)+u(1:end-2));
      unew(1)   = 0.0;
      unew(end) = 0.0;
  
     case {'beam-warming','bw'}
      unew(3:end) = (1-3/2*sigma+sigma^2/2)*u(3:end) ...
          + sigma*(2-sigma)*u(2:end-1) ...
          + sigma/2*(sigma-1)*u(1:end-2);
      unew(1)   = 0.0;
      unew(2)   = 0.0;
  
     case 'fromm'
      % interior stencil (as in your original code)
      unew(3:end-1) = sigma/4*(sigma-1)*u(1:end-3) ...
          + sigma/4*(5-sigma)*u(2:end-2) ...
          - 0.25*(sigma^2+3*sigma-4)*u(3:end-1) ...
          + sigma/4*(sigma-1)*u(4:end);
      unew(1) = 0.0;
      unew(2) = 0.0;

      uFic = ficVal(u,bcOrder); 
  
      j   = length(u);
      ujm2 = u(j-2);
      ujm1 = u(j-1);
      uj   = u(j);
      ujp1 = uFic;
      uLW = uj ...
          - 0.5*sigma*(ujp1 - ujm1) ...
          + 0.5*sigma^2*(ujp1 - 2*uj + ujm1);
  
      % Beam–Warming at j=M (a>0)
      uBW = uj ...
          - 0.5*sigma*(3*uj - 4*ujm1 + ujm2) ...
          + 0.5*sigma^2*(uj - 2*ujm1 + ujm2);
  
      % Fromm = average
      unew(end) = 0.5*(uLW + uBW);
  
     otherwise   % upwind/ftbs
      unew(2:end) = u(2:end) - sigma * (u(2:end) - u(1:end-1));
      unew(1) = 0.0;
  
    end
  
    % Calculate L1, L2 and Linf errors.
    uerr0(i) = norm(unew-uexact(x,t,a), inf);
    uerr1(i) = norm(unew-uexact(x,t,a), 1) / nx;
    uerr2(i) = norm(unew-uexact(x,t,a), 2) / sqrt(nx);
  

    if t >= ip*dtprint
      figure(1)
      plot(x, unew)
      xlabel('x'), ylabel('u')
      title(sprintf('%s (bc=%d) solution at time t=%9.4f', method, bcOrder, t))
      hold on, plot(xex, uex, 'r:'), hold off
      grid on, shg
      xlim([9 10]);
      ylim([-0.3 1.3]);
  
      pause(0.1)
      ip = ip + 1;
      uall(ip,:) = unew;
    end
  
    uold = u;   % save U^{n-1} for Leap-Frog only
    u = unew;   % save U^n
  end
  
  disp( ['U_max   = ', num2str(max(u))] );
  disp( ['L1-err  = ', num2str(uerr1(end))] );
  disp( ['L2-err  = ', num2str(uerr2(end))] );
  disp( ['Linf-err= ', num2str(uerr0(end))] );
  
  % Plot a sequence of profiles throughout the run.
  figure(2); clf
  nskip = 5;
  plot(x, uall(1:nskip:end,:));
  xlabel('x'), ylabel('u')
  grid on, shg
  print('-djpeg', sprintf('linadvUall_%s.jpg', runTag));
  
  % Plot the errors.
  figure(3); clf
  loglog(dt*(1:nt), [uerr0; uerr2; uerr1])
  xlabel('t'), ylabel('Error')
  legend('max-norm', 'L2-norm', 'L1-norm', 'Location', 'SouthEast')
  title(sprintf('Errors: %s (bc=%d)', method, bcOrder))
  axis tight
  grid on, shg
  print('-djpeg', sprintf('linadvErr_%s.jpg', runTag));
  
  end  % end linadv
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function uex = uexact( x, t, a )
  global ictype;
  u0shift = 2;
  switch ictype
   case 1
     uex = (abs(x-0.5-u0shift-a*t) < 0.5);
   case 2
    uex = exp(-20*(x-0.5-u0shift-a*t).^2);
   otherwise
    uex = 0.5*(1 + sin(2*pi*(x-1/4-u0shift-a*t))) .* ...
          (abs(x-0.5-u0shift-a*t)<0.5);
  end
  end  % end uexact
  
% ------------- MY FUNCTION ----------------
function uFic= ficVal(u,bcOrder)
  
  	uM=u(end);
  	uMm1=u(end-1);
  	switch bcOrder
    	case 0
      		uFic=uM;
  	case 1
      		uFic=2*uM-uMm1;
  	case 2
      		uMm2=u(end-2);
     	 	uFic=3*uM-3*uMm1+uMm2;
 	otherwise
      		error('ERR');
  	end
  
end
  

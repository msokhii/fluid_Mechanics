function runq5spotHW4_opt
    clc; close all;
    
    % Sample alpha values for optimization
    alphaVals = linspace(0,1,21);
    PoVals    = zeros(size(alphaVals));
    
    for k = 1:length(alphaVals)
        alpha = alphaVals(k);
        [PoVals(k), Tt, Ftot, SwFinal, PFinal, xplot, yplot, Grid, Tend] = simulateQuarterFiveSpot(alpha, false);
    
        fprintf('alpha = %.3f,  P_o = %.6f\n', alpha, PoVals(k));
    end
    
    % Find optimal alpha
    [PoMax, idx] = max(PoVals);
    alphaOpt = alphaVals(idx);
    
    fprintf('\nOptimal alpha = %.6f\n', alphaOpt);
    fprintf('Maximum total oil production P_o = %.6f\n', PoMax);
    
    % Plot P_o versus alpha
    figure;
    plot(alphaVals, PoVals, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
    grid on;
    xlabel('\alpha');
    ylabel('Total oil production P_o');
    title('Optimization of total oil production');
    hold on;
    plot(alphaOpt, PoMax, 'rs', 'MarkerSize', 10, 'LineWidth', 1.5);
    legend('P_o(\alpha)', sprintf('Optimal \\alpha = %.3f', alphaOpt), 'Location', 'best');
    
    % Re-run optimal case and display solution plots
    [PoOpt, Tt, Ftot, SwFinal, PFinal, xplot, yplot, Grid, Tend] = simulateQuarterFiveSpot(alphaOpt, true);
    
    fprintf('\nOptimal run complete:\n');
    fprintf('alphaOpt = %.6f,  P_o = %.6f\n', alphaOpt, PoOpt);
    
    end
    
    
    function [Po, Tt, Ftot, Sfinal, Pfinal, xplot, yplot, Grid, Tend] = simulateQuarterFiveSpot(alpha, doPlot)
    
    % ============================================================
    % Grid and physical parameters
    % ============================================================
    Grid.Nx = 64; 
    Dx = 2; 
    Grid.hx = Dx/Grid.Nx;          % doubled x-length
    
    Grid.Ny = 64; 
    Dy = 1; 
    Grid.hy = Dy/Grid.Ny;
    
    Grid.Nz = 1;  
    Dz = 1; 
    Grid.hz = Dz/Grid.Nz;
    
    N = Grid.Nx*Grid.Ny;                     
    Grid.V   = Grid.hx*Grid.hy*Grid.hz;        
    Grid.K   = ones(3,Grid.Nx,Grid.Ny,Grid.Nz);
    Grid.por = ones(Grid.Nx,Grid.Ny,Grid.Nz);
    
    % anisotropic permeability: Kx=10, Ky=1
    Grid.K(1,:,:,:) = 10;   % x-direction permeability
    Grid.K(2,:,:,:) = 1;    % y-direction permeability
    Grid.K(3,:,:,:) = 1;    % z-direction permeability
    
    xplot = linspace(Grid.hx/2, Dx-Grid.hx/2, Grid.Nx);
    yplot = linspace(Grid.hy/2, Dy-Grid.hy/2, Grid.Ny);
    
    Tend = 1.5;               % increased end time
    Fluid.vw = 1.0;           
    Fluid.vo = 10.0;          % oil viscosity increased
    Fluid.swc = 0.0; 
    Fluid.sor = 0.0;
    
    % ============================================================
    % Well setup
    % ============================================================
    % Linear indexing for (i,j) on an Nx-by-Ny grid in MATLAB:
    % index = i + (j-1)*Nx
    %
    % lower-left  = (1,1)
    % upper-right = (Nx,Ny)
    % upper-left  = (1,Ny)
    
    idxA = 1;                            % lower-left injector
    idxB = N;                            % upper-right producer
    idxC = 1 + (Grid.Ny-1)*Grid.Nx;      % upper-left producer
    
    Q = zeros(N,1);
    Q(idxA) = +1;          % injection well
    Q(idxB) = alpha - 1;   % production at upper-right
    Q(idxC) = -alpha;      % production at upper-left
    
    % Check total source strength = 0
    if abs(sum(Q)) > 1e-12
        error('Well rates do not sum to zero.');
    end
    
    % ============================================================
    % Initial condition and storage
    % ============================================================
    S = zeros(N,1);             % initially oil-filled reservoir
    
    nt = 50;                    
    dt = Tend/nt;
    
    Tt   = zeros(1,nt+1);       % time points
    Ftot = zeros(2,nt+1);       % row 1 = water total fraction, row 2 = oil total fraction
    
    % At t=0, S=0 everywhere, so f_w = 0 and f_o = 1 at both producers
    Ftot(:,1) = [0;1];
    
    if doPlot
        figure;
    end
    
    % ============================================================
    % Time stepping
    % ============================================================
    for it = 1:nt
        t = it*dt;
    
        [P,V] = Pres(Grid,S,Fluid,Q);      
        S     = Upstream(Grid,S,Fluid,V,Q,dt);
    
        % Flow fractions at upper-right producer B
        [lambdawB, lambdaoB] = RelPerm(S(idxB), Fluid);
        lambdaTotB = lambdawB + lambdaoB;
        fwB = lambdawB / lambdaTotB;
        foB = lambdaoB / lambdaTotB;
    
        % Flow fractions at upper-left producer C
        [lambdawC, lambdaoC] = RelPerm(S(idxC), Fluid);
        lambdaTotC = lambdawC + lambdaoC;
        fwC = lambdawC / lambdaTotC;
        foC = lambdaoC / lambdaTotC;
    
        % Total fractions from the two production wells
        fwTOT = (1-alpha)*fwB + alpha*fwC;
        foTOT = (1-alpha)*foB + alpha*foC;
    
        Tt(it+1)     = t;
        Ftot(:,it+1) = [fwTOT; foTOT];
    
        if doPlot
            % Plot #1: saturation
            subplot(2,2,1)
            contourf(xplot, yplot, reshape(S,Grid.Nx,Grid.Ny), 11, 'k');
            axis square
            caxis([0 1]);
            colorbar
            title(sprintf('Water saturation, t = %.3f', t))
    
            % Plot #2: pressure
            subplot(2,2,2)
            contourf(xplot, yplot, reshape(P,Grid.Nx,Grid.Ny), 11, 'k');
            axis square
            colorbar
            title(sprintf('Pressure, t = %.3f', t))
    
            % Plot #3: total production fractions
            subplot(2,2,[3,4])
            plot(Tt, Ftot(2,:), 'b-', 'LineWidth', 1.5); hold on;
            plot(Tt, Ftot(1,:), 'r-', 'LineWidth', 1.5); hold off;
            axis([0 Tend -0.05 1.05]);
            grid on
            legend('Oil cut','Water cut','Location','West');
            title(sprintf('Total flow fractions, \\alpha = %.3f', alpha))
            xlabel('t')
            ylabel('fraction')
            drawnow
        end
    end
    
    % ============================================================
    % Total oil produced: trapezoidal rule
    % ============================================================
    Po = trapz(Tt, Ftot(2,:));
    
    Sfinal = S;
    Pfinal = P;
    
    end
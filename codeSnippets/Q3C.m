% Parameters
a=1;
muVals=[0.002,0.005];
xLeft=-5;
xRight=10;
T=4;
dxVals=[1/10,1/15,1/20];
cflTarget=0.9;
xexPlot = linspace(xLeft, xRight, 2500).';   
quadRes = 3500;

for idx=1:numel(muVals)
    mu=muVals(idx);
    fprintf('\n MU VAL -> %.4f\n', mu);
    resStruct=struct([]);
    figure('Name',sprintf('Final profiles for MU = %.4f',mu));
    tiledlayout(1, 3,'Padding','compact','TileSpacing','compact');
    for idx2=1:numel(dxVals)
        dx=dxVals(idx2);
        x=(xLeft:dx:xRight).';
        [dt,Nt]=getBestT(dx,a,T,cflTarget);
        V=a*dt/dx;
        sig=mu*dt/dx^2; 
        vonCond=(V+2*sig<=1);
        tSnaps=[0;1;2;4];
        [uT,U_snaps,tAct]=mySolver(x,T,dt,a,mu,tSnaps);
		uexT = compUEX(x, T, a, mu, quadRes);
		errL2 = sqrt(dx * sum((uT - uexT).^2));
		resStruct(idx2).dx=dx;
        	resStruct(idx2).dt=dt;
        	resStruct(idx2).Nt=Nt;
        	resStruct(idx2).V=V;
        	resStruct(idx2).sig=sig;
        	resStruct(idx2).Vp2s=V+2*sig;
        	resStruct(idx2).vonCond=vonCond;
        	resStruct(idx2).errL2=errL2;
        	fprintf('Delta X VAL=%.6f | DELTA T VAL=%.8f | NT=%4d | V=%.4f | SIGMA=%.4f | pred:%s | L2err=%.3e\n', ...
        	dx, dt, Nt, V, sig,tf(vonCond), errL2);

        	nexttile;
       		plot(x, uT, 'LineWidth', 1.2); 
		hold on;
        	uexFine = compUEX(xexPlot, T, a, mu, quadRes);
        	plot(xexPlot, uexFine, 'r--', 'LineWidth', 1.2); hold off;
        	grid on;
        	xlabel('x'); 
		ylabel('u');
        	title(sprintf('\\Deltax=%.4g | V=%.3f', dx, V));
        	legend('Computed Solution', 'Exact Solution', 'Location', 'best');
    	end

    dxFine = dxVals(end);
    [dtFixed, NtFixed] = getBestT(dxFine, a, T, cflTarget);
    fprintf('Fixed dt = %.10f | Nt = %d\n', dtFixed, NtFixed);

    errs  = zeros(size(dxVals));
    Vvals = zeros(size(dxVals));
    Vp2s  = zeros(size(dxVals));

    for idx3 = 1:numel(dxVals)
        dx = dxVals(idx3);
        x  = (xLeft:dx:xRight).';

        V   = a*dtFixed/dx;
        sig = mu*dtFixed/dx^2;

        uT   = mySolver(x, T, dtFixed, a, mu, []); % no snapshots
        uexT = compUEX(x, T, a, mu, quadRes);

        errs(idx3) = sqrt(dx * sum((uT - uexT).^2));
        Vvals(idx3) = V;
        Vp2s(idx3)  = V + 2*sig;

        fprintf('dx=%.6f | V=%.4f | sig=%.4f | V+2sig=%.4f | pred:%s | L2err=%.3e\n', ...
            dx, V, sig, V + 2*sig, tf(V + 2*sig <= 1), errs(idx3));
    end

    figure('Name', sprintf('Error vs dx (fixed dt), mu=%.4f', mu));
    loglog(dxVals, errs, 'o-', 'LineWidth', 1.2);
    grid on;
    xlabel('\Deltax');
    ylabel('L^2 error at t=4');
    title(sprintf('Error vs \\Deltax (mu=%.4g), fixed dt from finest grid', mu));
    p = polyfit(log(dxVals(:)), log(errs(:)), 1);
    fprintf('Observed slope (order) ~ %.3f\n', p(1));

function [dt, Nt] = getBestT(dx, a, T, cflTarget)
    dtTarget = cflTarget * dx / a;
    Nt = ceil(T / dtTarget);
    dt = T / Nt;
end

function [uT, U_snaps] = mySolver(x, T, dt, a, mu, tSnaps)
    dx = x(2) - x(1);
    N  = numel(x);
    Nt = round(T / dt);
    V   = a*dt/dx;
    sig = mu*dt/dx^2;
    u = max(1 - abs(x), 0);
    u(1) = 0; u(end) = 0;
    uNew = u;

    for n = 1:Nt
        uNew(2:end-1) = u(2:end-1) ...
            - V  * (u(2:end-1) - u(1:end-2)) ...
            + sig * (u(3:end) + u(1:end-2) - 2*u(2:end-1));
        uNew(1) = 0;
        uNew(end) = 0;

        u = uNew;
        if ~isempty(snapIdx)
            hit = find(snapIdx == n, 1);
            if ~isempty(hit)
                U_snaps(:, hit) = u;
            end
        end
    end

    uT = u;
end

function uex=compUEX(x,t,a,mu,Ny)
    if t==0
        uex=max(1-abs(x),0);
        return;
    end
    y =linspace(-1,1,Ny);
    u0=max(1-abs(y),0);
    X=x(:); 
    Y=y(1,:);    
    res=exp(-(X-Y-a*t).^2/(4*mu*t));
    compRes=trapz(y,res.*u0,2);
    uex=(1/sqrt(4*pi*mu*t))*compRes;
    uex=reshape(uex,size(x));
end

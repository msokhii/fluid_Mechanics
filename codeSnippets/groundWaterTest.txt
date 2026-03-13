function opVal=gwCompare   
    clc;    
    T=2.0;
    deltaX=[0.04,0.02,0.01,0.005];
    ele=numel(deltaX); % This is 4.
    % L^2 error + cpu time + total time steps + true time steps.
    % These are 4x1 arrays initialized with 0's.
    errorFT=zeros(ele,1);
    errorCN=zeros(ele,1);
    cpuTimeFT=zeros(ele,1);
    cpuTimeCN=zeros(ele,1);
    compTimeStepFT=zeros(ele,1);
    compTimeStepCN=zeros(ele,1);
    trueTimeStepFT=zeros(ele,1);
    trueTimeStepCN=zeros(ele,1);
    
    for iter=1:ele
        eleChosen=deltaX(iter);
        cellSize=round(1/eleChosen);
        fprintf('Delta X -> %.4f with cell size -> %d\n',eleChosen,cellSize);   
        [errorFT(iter),cpuTimeFT(iter),compTimeStepFT(iter),trueTimeStepFT(iter)]=mySolver(eleChosen, T, 'FT');
        [errorCN(iter),cpuTimeCN(iter),compTimeStepCN(iter),trueTimeStepCN(iter)]=mySolver(eleChosen, T, 'CN');
    end
    
    % These are 4x1 arrays initialized with NaN's.
    orderFT=nan(ele,1);
    orderCN=nan(ele,1);
    % Checking convergence rate.
    for iter=2:ele
        orderFT(iter)=log(errorFT(iter-1)/errorFT(iter))/log(2);
        orderCN(iter)=log(errorCN(iter-1)/errorCN(iter))/log(2);
    end
    
    % Bookkeeping stuff for printing summarized results table.
    opVal=table(deltaX(:),trueTimeStepFT,compTimeStepFT,errorFT,cpuTimeFT,orderFT, ...
            trueTimeStepCN,compTimeStepCN,errorCN,cpuTimeCN,orderCN, ...
            'VariableNames',{'DeltaX','True_Steps_FT','Total_Steps_FT','error_FT', ...
            'CPU_Time_FT','conv_Rate_FT','True_Steps_CN','Total_Steps_CN', ...
            'error_CN','CPU_Time_CN','conv_Rate_CN'});
    
    fprintf('\nResults: \n');
    disp(opVal);
    
    for iter=1:ele
        fprintf(['%.4f & %.6e & %.6f & %.3f & %.6e & %.3f \\\\\n'], ...
        deltaX(m),errorFT(m),orderFT(m),cpuTimeFT(m),errorCN(m),cpuTimeCN(m));
    end
end
    
function [err,endTime,adjStepSize,dt]=mySolver(cellSize,T,method)
    % Cell centered grid parameters.
    N=round(1/cellSize);
    x=((1:N)-0.5)*cellSize;
    y=((1:N)-0.5)*cellSize;
    [X,Y]=ndgrid(x,y);
    % Initial conditions.
    uNot=exU(X,Y,0);
    U=uNot(:);
    % A is the sparse matrix that is required to solve the spatial part of the PDE.
    A=mkMatrix(N,cellSize,x,y);
    switch upper(method)
        case 'FIT'
            targetSize=cellSize^2/12;
        case 'CN'
            targetSize=cellSize;
    end
    adjStepSize=ceil(T/targetSize);
    dt=T/adjStepSize;
    % Terms and boundary conditions.
    alphaVal=1.0; 
    decayVal=exp(-dt);
    startTimer=cputime;
    switch upper(method)
        case 'FIT'
            for n=1:adjStepSize
                U=U+dt*(A*U+alphaVal*fAT0);
                alphaVal=alphaVal*decayVal;
            end
        case 'CN'
            I=speye(N*N);
            leftDeltaX=I-0.5*dt*A;
            rightDeltaX=I+0.5*dt*A;
            fileChosen=exist('decomposition','file')==2;
            if fileChosen
                D=decomposition(leftDeltaX,'lu');
            end
            for n=1:adjStepSize
                alphaNext=alphaVal*decayVal;
                rdeltaX=rightDeltaX*U+0.5*dt*(alphaVal+alphaNext)*fAT0;
                if fileChosen
                    U=D\rdeltaX;
                else
                    U=leftDeltaX\rdeltaX;
                end
                    alphaVal=alphaNext;
            end
    end
    
    endTime=cputime-startTimer;
    % This is our exact solution + error at time T.
    Uex=exU(X,Y,T);
    err=sqrt(cellSize^2*sum((U-Uex(:)).^2));
end
        
function retMatrix=mkMatrix(N,cellSize,x,y) 
    maxNNZ=5*N*N;
    % These are (5*N*N)x1 arrays initialized with 0's.
    I=zeros(maxNNZ,1);
    J=zeros(maxNNZ,1);
    S=zeros(maxNNZ,1);
    k=0;
    invh2=1/cellSize^2;
    for j=1:N
        yj=y(j);
        for i=1:N
            xi=x(i);
            p=idx(i,j,N);
            % Coefficient: a(x,y)=1+x+y.
            aW=1+(xi-0.5*cellSize)+yj;
            aE=1+(xi+0.5*cellSize)+yj;
            aS=1+xi+(yj-0.5*cellSize);
            aN=1+xi+(yj+0.5*cellSize);
            diagVal=0.0;
    
            % West boundary.
            if i>1
                k=k+1;
                I(k)=p; 
                J(k)=idx(i-1,j,N); 
                S(k)=aW*invh2;
                diagVal=diagVal-aW*invh2;
            else
                % Half-cell Dirichlet.
                diagVal=diagVal-2*aW*invh2;
            end
            
            % East boundary.
            if i<N
                k=k+1;
                I(k)=p; 
                J(k)=idx(i+1,j,N); 
                S(k)=aE*invh2;
                diagVal=diagVal-aE*invh2;
            else
                diagVal=diagVal-2*aE*invh2;
            end
            
            % South boundary.
            if j>1
                k=k+1;
                I(k)=p; 
                J(k)=idx(i,j-1,N); 
                S(k)=aS*invh2;
                diagVal=diagVal-aS*invh2;
            else
                diagVal=diagVal-2*aS*invh2;
            end
            
            % North boundary.
            if j<N
                k=k+1;
                I(k)=p; 
                J(k)=idx(i,j+1,N); 
                S(k)=aN*invh2;
                diagVal=diagVal-aN*invh2;
            else
                diagVal=diagVal-2*aN*invh2;
            end
    
            % Diagonal boundary.
            k=k+1;
            I(k)=p; 
            J(k)=p; 
            S(k)=diagVal;
        end
    end
        
        % Returns A = sparse matrix.
        retMatrix=sparse(I(1:k),J(1:k),S(1:k),N*N,N*N);
end
    
function uExact=exU(X,Y,t)
    uExact=exp(-t).*sin(pi*X).*(1-exp(-Y));
end

% phi: 2D ID -> Vector_p
function map=idx(i, j, N)
    map=i+(j-1)*N;
end

% Run.
opVal=gwCompare;
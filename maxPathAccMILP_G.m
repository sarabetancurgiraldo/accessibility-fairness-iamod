function maxPathAccMILP_G(G,B,pc_unique,Xfast,Xslow,D,nOD,R_selector,Nmin, ...
                        population_region,str_save,alpha,Tmax,nArcs,nCar)
%% Parameters
M               = 1e2;

%% Variables
t               = G.Edges.Weight;
nR              = length(pc_unique); % # regions

% AV(car) layer variables 
arcsCar         = find(G.Edges.Type == 1);
nCarArcs        = sum(G.Edges.Type == 1);
Bcar            = B(:,arcsCar);

% epsilon - Time above threshold matrix
% Efast full graph         (fast) 
% Eslow graph w/out cars   (slow) 
tfast           = t'*Xfast;
tslow           = t'*Xslow;
Efast           = (max(0, tfast - Tmax));  
Eslow           = (max(0, tslow - Tmax));  
% Efast           = max(0, t'*Xfast - Tmax);
% Eslow           = max(0, t'*Xslow - Tmax);

% Variable definition
xR              = sdpvar(nCarArcs, 1,'full');
Ffast           = sdpvar(1, nOD,'full');
Fslow           = sdpvar(1, nOD,'full');
b               = binvar(1, nOD,'full'); %definition of binary variable
epsilon         = sdpvar(nR, 1, 'full');

% Matrix form for od-pair optimization
% X_m = ffast_m*Xfast_m + fslow_m*Xslow_m
% X = [X_1 ... X_m]
% Ffast = dim_f*[ffast_1; ...; ffast_m] (dim: nArcs x nOD)
dim_f           = ones(nArcs,1);
X               = dim_f*Ffast.*Xfast + dim_f*Fslow.*Xslow;
N               = R_selector * b'; % N means number of reachable destinations per every region

% Constraints
Cons            = [Bcar*(sum(X(arcsCar,:),2)+xR)                == 0;
                   t(arcsCar)'*(sum(X(arcsCar,:),2)+xR)         <= nCar;
                   Ffast + Fslow                                == alpha;
                   xR                                           >= 0;
                   Ffast                                        >= 0;
                   Fslow                                        >= 0
                   (Ffast.*Efast+Fslow.*Eslow)./alpha           <= (1-b)*M; %Remove -Tmax
                   epsilon                                      >= (Nmin - N)/Nmin;   %here epsilon would be delta Nr
                   epsilon                                      >= 0]; 


%% Optimization

% Objective
Obj             = (population_region' * epsilon + ... 
                   1e2*t'*X*ones(size(D,2),1)/sum(abs(D),"all"))...
                   /(sum(population_region));%*Nmin);

options         = sdpsettings('verbose', 1, ...
                              'solver', 'gurobi', ...
                              'showprogress', 1);

% Save variables in object
sol_pathAccMILP         = optimize(Cons, Obj, options);
sol_pathAccMILP.X       = value(X);
sol_pathAccMILP.xR      = value(xR);
sol_pathAccMILP.Ffast   = value(Ffast);
sol_pathAccMILP.Fslow   = value(Fslow);
sol_pathAccMILP.Efast   = Efast;
sol_pathAccMILP.Eslow   = Eslow;
sol_pathAccMILP.b       = value(b);
xRfull                  = zeros(nArcs,1);
xRfull(arcsCar)         = xR;
sol_pathAccMILP.xR      = xRfull;
AFI_approx              = (1-value(b))*sum(abs(D),1)'/sum(sum(abs(D)));
sol_pathAccMILP.AFI_approx = AFI_approx;
sol_pathAccMILP.tfast   = tfast;
sol_pathAccMILP.tslow   = tslow;
sol_pathAccMILP.N        = value(N);
sol_pathAccMILP.epsilon  = value(epsilon);
sol_pathAccMILP.Obj      = value(Obj);

save(str_save,"sol_pathAccMILP");

end
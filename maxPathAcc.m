function sol_pathAcc = maxPathAcc(Tmax,nCar,str_save,G,B,D,nArcs, ...
                                       Xfast,Xslow,nOD, ...
                                       population_region,R_selector,alpha)

% Define Parameters
t               = G.Edges.Weight;

% AV(car) layer variables 
arcsCar         = find(G.Edges.Type == 1);
nCarArcs        = sum(G.Edges.Type == 1);
Bcar            = B(:,arcsCar);

% epsilon - Time above threshold matrix
% Efast full graph         (fast) 
% Eslow graph w/out cars   (slow) 
tfast           = t'*Xfast;
tslow           = t'*Xslow;
Efast           = (max(0, tfast - Tmax)/Tmax).^2;
Eslow           = (max(0, tslow - Tmax)/Tmax).^2;


% Optimization

% Variable definition
xR              = sdpvar(nCarArcs, 1,'full');
Ffast           = sdpvar(1, nOD,'full');
Fslow           = sdpvar(1, nOD,'full');

% Matrix form for od-pair optimization
% X_m = ffast_m*Xfast_m + fslow_m*Xslow_m
% X = [X_1 ... X_m]
% Ffast = dim_f*[ffast_1; ...; ffast_m] (dim: nArcs x nOD)
dim_f           = ones(nArcs,1);
X               = dim_f*Ffast.*Xfast + dim_f*Fslow.*Xslow;

% Constraints
Cons            = [Bcar*(sum(X(arcsCar,:),2)+xR)                == 0;
                   t(arcsCar)'*(sum(X(arcsCar,:),2)+xR)         <= nCar;
                   Ffast + Fslow                                == alpha;
                   xR                                           >= 0;
                   Ffast                                        >= 0;
                   Fslow                                        >= 0]; 

% Objective
Ur = R_selector*((Ffast.*Efast+Fslow.*Eslow)')./(R_selector*alpha');
% Ur = R_selector*((Ffast.*Efast.^2+Fslow.*Eslow.^2)')./(R_selector*alpha');
Obj             = (population_region' * Ur + ... 
                   1e-2*t'*X*ones(size(D,2),1)/sum(abs(D),"all"))...
                   /(sum(population_region)); 


options         = sdpsettings('verbose', 1, ...
                              'solver', 'gurobi', ...
                              'showprogress', 1);

sol_pathAcc         = optimize(Cons, Obj, options);
sol_pathAcc.X       = value(X);
sol_pathAcc.xR      = value(xR);
sol_pathAcc.tfast   = tfast;
sol_pathAcc.tslow   = tslow;
sol_pathAcc.Ffast   = value(Ffast);
sol_pathAcc.Fslow   = value(Fslow);
Ur                  = value(Ur);
sol_pathAcc.Ur      = Ur;
sol_pathAcc.Efast   = Efast;
sol_pathAcc.Eslow   = Eslow;
sol_pathAcc.Obj     = value(Obj);

UnfIndR             = population_region .* Ur / sum(population_region);
sol_pathAcc.UnfIndR = UnfIndR;

save(str_save,"sol_pathAcc");

end
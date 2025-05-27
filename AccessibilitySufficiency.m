function AccessibilitySufficiency(G,B,pc_unique,X_Iamod,X_amod,D,nOD, ...
                                  R_selector,Nsuff,population_region, ...
                                  str_save,alpha,Tsuff,nArcs,nCar)
%% Parameters
M               = 1e2;

%% Variables
t               = G.Edges.Weight;
nR              = length(pc_unique); % # regions

% AV(car) layer variables 
arcsCar         = find(G.Edges.Type == 1);
nCarArcs        = sum(G.Edges.Type == 1);
Bcar            = B(:,arcsCar);

% epsilon - Time above threshold (matrix)
% E_Iamod full graph
% E_amod graph w/out cars 
t_Iamod         = t'*X_Iamod;
t_amod          = t'*X_amod;
E_Iamod         = (max(0, t_Iamod - Tsuff));
E_amod          = (max(0, t_amod - Tsuff));

% Variable definition
xR              = sdpvar(nCarArcs, 1,'full');
F_Iamod         = sdpvar(1, nOD,'full');
F_amod          = sdpvar(1, nOD,'full');
b               = binvar(1, nOD,'full'); %definition of binary variable
u_r             = sdpvar(nR, 1, 'full');

% Matrix form for od-pair optimization
% X_m = f_Iamod_m*X_Iamod_m + f_amod_m*X_amod_m
% X = [X_1 ... X_m]
% F_Iamod = dim_f*[f_Iamod_1; ...; f_Iamod_m] (dim: nArcs x nOD)
dim_f           = ones(nArcs,1);
X               = dim_f*F_Iamod.*X_Iamod + dim_f*F_amod.*X_amod;
N               = R_selector * b'; % N means number of reachable destinations per every region

% Constraints
Cons            = [Bcar*(sum(X(arcsCar,:),2)+xR)                == 0;
                   t(arcsCar)'*(sum(X(arcsCar,:),2)+xR)         <= nCar;
                   F_Iamod + F_amod                             == alpha;
                   xR                                           >= 0;
                   F_Iamod                                      >= 0;
                   F_amod                                       >= 0
                   (F_Iamod.*E_Iamod+F_amod.*E_amod)./alpha     <= (1-b)*M; 
                   u_r                                          >= ((Nsuff - N)/Nsuff).^2;   
                   u_r                                          >= 0]; 


%% Optimization

% Objective
Obj             = (population_region' * u_r + ... 
                   1e2*t'*X*ones(size(D,2),1)/sum(abs(D),"all"))...
                   /(sum(population_region));

options         = sdpsettings('verbose', 1, ...
                              'solver', 'gurobi', ...
                              'showprogress', 1);

% Save variables in object
sol_AccSuff             = optimize(Cons, Obj, options);
sol_AccSuff.X           = value(X);
sol_AccSuff.xR          = value(xR);
sol_AccSuff.F_Iamod     = value(F_Iamod);
sol_AccSuff.F_amod      = value(F_amod);
sol_AccSuff.E_Iamod     = E_Iamod;
sol_AccSuff.E_amod      = E_amod;
sol_AccSuff.b           = value(b);
xRfull                  = zeros(nArcs,1);
xRfull(arcsCar)         = xR;
sol_AccSuff.xR          = xRfull;
AFI_approx              = (1-value(b))*sum(abs(D),1)'/sum(sum(abs(D)));
sol_AccSuff.AFI_approx  = AFI_approx;
sol_AccSuff.t_Iamod     = t_Iamod;
sol_AccSuff.t_amod      = t_amod;
sol_AccSuff.N           = value(N);
sol_AccSuff.u_r         = value(u_r);
sol_AccSuff.Obj         = value(Obj);

save(str_save,"sol_AccSuff");

end
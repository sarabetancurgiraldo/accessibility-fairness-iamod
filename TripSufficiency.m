function sol_Tripsuff = TripSufficiency(Tsuff,nCar,str_save,G,B,D, ...
                                        nArcs,X_Iamod,X_amod,nOD, ...
                                        population_region,R_selector,alpha)

% Define Parameters
t               = G.Edges.Weight;

% AV(car) layer variables 
arcsCar         = find(G.Edges.Type == 1);
nCarArcs        = sum(G.Edges.Type == 1);
Bcar            = B(:,arcsCar);

% epsilon - Time above threshold (matrix)
% E_Iamod full graph
% E_amod graph w/out cars 
t_Iamod         = t'*X_Iamod;
t_amod          = t'*X_amod;
E_Iamod         = (max(0, t_Iamod - Tsuff)/Tsuff).^2;
E_amod          = (max(0, t_amod - Tsuff)/Tsuff).^2;


% Optimization

% Variable definition
xR              = sdpvar(nCarArcs, 1,'full');
F_Iamod         = sdpvar(1, nOD,'full');
F_amod          = sdpvar(1, nOD,'full');

% Matrix form for od-pair optimization
% X_m = f_Iamod_m*X_Iamod_m + f_amod_m*X_amod_m
% X = [X_1 ... X_m]
% F_Iamod = dim_f*[f_Iamod_1; ...; f_Iamod_m] (dim: nArcs x nOD)
dim_f           = ones(nArcs,1);
X               = dim_f*F_Iamod.*X_Iamod + dim_f*F_amod.*X_amod;

% Constraints
Cons            = [Bcar*(sum(X(arcsCar,:),2)+xR)                == 0;
                   t(arcsCar)'*(sum(X(arcsCar,:),2)+xR)         <= nCar;
                   F_Iamod + F_amod                             == alpha;
                   xR                                           >= 0;
                   F_Iamod                                      >= 0;
                   F_amod                                       >= 0]; 

% Objective
Ur = R_selector*((F_Iamod.*E_Iamod+F_amod.*E_amod)')./(R_selector*alpha');
Obj             = (population_region' * Ur + ... 
                   1e-2*t'*X*ones(size(D,2),1)/sum(abs(D),"all"))...
                   /(sum(population_region)); 


options         = sdpsettings('verbose', 1, ...
                              'solver', 'gurobi', ...
                              'showprogress', 1);

sol_Tripsuff            = optimize(Cons, Obj, options);
sol_Tripsuff.X          = value(X);
sol_Tripsuff.xR         = value(xR);
sol_Tripsuff.t_Iamod    = t_Iamod;
sol_Tripsuff.t_amod     = t_amod;
sol_Tripsuff.F_Iamod    = value(F_Iamod);
sol_Tripsuff.F_amod     = value(F_amod);
Ur                      = value(Ur);
sol_Tripsuff.Ur         = Ur;
sol_Tripsuff.E_Iamod    = E_Iamod;
sol_Tripsuff.E_amod     = E_amod;
sol_Tripsuff.Obj        = value(Obj);

UnfIndR                 = population_region .* Ur / sum(population_region);
sol_Tripsuff.UnfIndR    = UnfIndR;

save(str_save,"sol_Tripsuff");

end
function avgAcc_reg(nCar,Tmax,str_save,G,B,D,nArcs)

% Define parameters
arcsCar                 = find(G.Edges.Type == 1);
nCarArcs                = sum(G.Edges.Type == 1);
Bcar                    = B(:,arcsCar);
t                       = G.Edges.Weight;

% Define variables
X                       = sdpvar(nArcs, size(D,2),'full');
xR                      = sdpvar(nCarArcs, 1,'full');
epsilon                 = sdpvar(size(D,2),1,'full');

y_i                     = (t'*X)'./(sum(abs(D),1)'/2);

Cons                    = [B*X                                      == D;
                           Bcar*(sum(X(arcsCar,:),2)+xR)            == 0;
                           t(arcsCar)' * (sum(X(arcsCar,:),2)+xR)   <= nCar;
                           X                                        >= 0;
                           xR                                       >= 0;
                           epsilon                                  >= 0;
                           epsilon                                  >= (y_i-Tmax)/Tmax];

Obj                     = (epsilon.^2'*sum(abs(D),1)' + ...
                           1e-4*t'*X*ones(size(D,2),1) + ...
                           60 * epsilon')... 
                           /sum(abs(D),"all");

options                 = sdpsettings('verbose', 1, ...
                                      'solver', 'gurobi', ...
                                      'showprogress', 1, ...
                                      'debug',1);
options.gurobi.FeasibilityTol = 1e-9;
options.gurobi.OptimalityTol = 1e-9;
options.gurobi.BarConvTol = 1e-9;
options.gurobi.IntFeasTol = 1e-9;
options.gurobi.PSDTol = 1e-9;
options.gurobi.TuneTimeLimit = 0;

sol_avgAcc                  = optimize(Cons, Obj, options);
X                           = value(X);
sol_avgAcc.X                = X;
xR                          = value(xR);
xRfull                      = zeros(nArcs,1);
xRfull(arcsCar)             = xR;
sol_avgAcc.xR               = xRfull;
epsilon                     = value(epsilon);
sol_avgAcc.epsilon          = epsilon.^2;
AFI_approx = value(epsilon)'*sum(abs(D),1)'/(Tmax*sum(sum(sum(abs(D)))));
sol_avgAcc.AFI_approx       = AFI_approx;
sol_avgAcc.Obj              = value(Obj);

% Save
save(str_save,"sol_avgAcc");


end
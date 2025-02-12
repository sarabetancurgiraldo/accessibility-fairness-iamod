close all; clear; clc;

pwd
addpath(genpath("/home/mech001/20222295/Sara/YALMIP-master"));
fclose('all');

load('model/data_g.mat');
load("model/data_shortPaths.mat");
% load('output/TT_AFI.mat','minTT','pathAcc')

%% Variables

% nCar = 4e3; 
% nCarRange = [1e3 2e3 4e3];
nCarRange = [ 3e3 ]; 
Tmax = 20/60;
Nmin = 35;

% alpha matrix - # of trips per hour for each od-pair 
alpha           = sum(abs(D),1)/2;
t               = G.Edges.Weight;

% nOD             = 5;
% D               = D(:,1:nOD);
% Xfast           = Xfast(:,1:nOD);
% Xslow           = Xslow(:,1:nOD);
% R_selector      = R_selector(:,1:nOD);
% alpha           = alpha(:,1:nOD);

%% Create object with data
nC = length(nCarRange);

MIQP = zeros(1,nC,3);

for i_nCar = 1:nC
nCar = nCarRange(i_nCar); 

% optimizations

% MIQP
str_save = sprintf('output/nCar/%d/MIQP.mat',nCar);
opt_MIQP(G,B,pc_unique,Xfast,Xslow,D,nOD,R_selector,Nmin, ...
         population_region,str_save,alpha,Tmax,nArcs,nCar)

% path flow allocation

% MIQP
load(sprintf('output/nCar/%d/MIQP.mat',nCar));
X_matrix = sol_MIQP.X;
t_AvgPathMIQP = (sol_MIQP.Ffast.*sol_MIQP.tfast + ...
                 sol_MIQP.Fslow.*sol_MIQP.tslow)./...
                 (sol_MIQP.Ffast+sol_MIQP.Fslow);
epsilonPathMIQP = (max(0,t_AvgPathMIQP-Tmax)/Tmax).^2;
epsilonPathMIQP = epsilonPathMIQP(1:nOD);
fp_save = sprintf('output/nCar/%d/path_flows_MIQP.mat',nCar);
[MIQP(1,i_nCar,1),...
 MIQP(1,i_nCar,2),...
 MIQP(1,i_nCar,3)] = path_flows_Leo(Tmax,X_matrix,epsilonPathMIQP',fp_save,D,B,G);

% Heatmap

% MIQP
load(sprintf('output/nCar/%d/MIQP.mat',nCar))
Efast = max(0,60*(sol_MIQP.tfast-Tmax)).^2;
Eslow = max(0,60*(sol_MIQP.tslow-Tmax)).^2;
epsilonPathMIQP = (sol_MIQP.Ffast.*Efast + ...
                   sol_MIQP.Fslow.*Eslow)/sum(alpha);
fp_load = sprintf('output/nCar/%d/path_flows_MIQP.mat',nCar);
fp_save = sprintf('output/nCar/%d/AFI_heatmap_MIQP.mat',nCar);
AFI_heatmap_sq(Tmax,fp_load,fp_save,epsilonPathMIQP',D,true)

end

str_save = sprintf('output/nCar/TT_AFI_MIQP_%d.mat',nCar);
save(str_save,'MIQP');



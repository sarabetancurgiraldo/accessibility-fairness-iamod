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
nCarRange = [1e3 3e3 4e3]; 
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

minTT = zeros(1,nC,3);
avgAcc = zeros(1,nC,3);
pathAcc = zeros(1,nC,3);
pathAccMILP = zeros(1,nC,3);

for i_nCar = 1:nC
nCar = nCarRange(i_nCar); 

% optimizations

% minTT
str_save = sprintf('output/nCar/%d/minTT.mat',nCar);
minTravelTime(nCar,G,B,nArcs,D,str_save);

% avgAcc
str_save = sprintf('output/nCar/%d/avgAcc.mat',nCar);
maxAverageAcc(nCar,Tmax,str_save,G,B,D,nArcs);

% pathAcc 
str_save = sprintf('output/nCar/%d/pathAcc.mat',nCar);
maxPathAcc(Tmax,nCar,str_save,G,B,D,nArcs,Xfast,Xslow,nOD, ...
           population_region,R_selector,alpha);

% % pathAccMILP
% str_save = 'output/pathAccMILP.mat';
% maxPathAccMILP(G,B,pc_unique,Xfast,Xslow,D,nOD,R_selector,Nmin, ...
%                population_region,str_save,alpha,Tmax,nArcs,nCar)
% pathAccMILP_G
str_save = sprintf('output/nCar/%d/pathAccMILP.mat',nCar);
maxPathAccMILP_G(G,B,pc_unique,Xfast,Xslow,D,nOD,R_selector,Nmin, ...
                 population_region,str_save,alpha,Tmax,nArcs,nCar)

% path flow allocation

% minTT
load(sprintf('output/nCar/%d/minTT.mat',nCar));
X_matrix = sol_mintt.X;
X_matrix = X_matrix(:,1:nOD);
epsilonTT = (max(0,(2*(t'*X_matrix)'./sum(abs(D),1)')-Tmax)/Tmax).^2;
fp_save = sprintf('output/nCar/%d/path_flows_minTT.mat',nCar);
[minTT(1,i_nCar,1), ...
 minTT(1,i_nCar,2), ...
 minTT(1,i_nCar,3)] = path_flows_Leo(Tmax,X_matrix,epsilonTT,fp_save,D,B,G);

% avgAcc
load(sprintf('output/nCar/%d/avgAcc.mat',nCar));
X_matrix = sol_avgAcc.X;
X_matrix = X_matrix(:,1:nOD);
epsilonAvg = sol_avgAcc.epsilon;
epsilonAvg = epsilonAvg(1:nOD);
fp_save = sprintf('output/nCar/%d/path_flows_avgAcc.mat',nCar);
[avgAcc(1,i_nCar,1), ...
 avgAcc(1,i_nCar,2), ...
 avgAcc(1,i_nCar,3)] = path_flows_Leo(Tmax,X_matrix,epsilonAvg,fp_save,D,B,G);

% pathAcc 
load(sprintf('output/nCar/%d/pathAcc.mat',nCar))
X_matrix = sol_pathAcc.X;
X_matrix = X_matrix(:,1:nOD);
t_AvgPath = (sol_pathAcc.Ffast.*sol_pathAcc.tfast + ...
             sol_pathAcc.Fslow.*sol_pathAcc.tslow)./...
             (sol_pathAcc.Ffast+sol_pathAcc.Fslow);
epsilonPath = (max(0,t_AvgPath-Tmax)/Tmax).^2;
epsilonPath = epsilonPath(1:nOD);
fp_save = sprintf('output/nCar/%d/path_flows_pathAcc.mat',nCar);
[pathAcc(1,i_nCar,1), ...
 pathAcc(1,i_nCar,2), ...
 pathAcc(1,i_nCar,3)] = path_flows_Leo(Tmax,X_matrix,epsilonPath',fp_save,D,B,G);

% pathAccMILP
load(sprintf('output/nCar/%d/pathAccMILP.mat',nCar));
X_matrix = sol_pathAccMILP.X;
t_AvgPathMILP = (sol_pathAccMILP.Ffast.*sol_pathAccMILP.tfast + ...
                 sol_pathAccMILP.Fslow.*sol_pathAccMILP.tslow)./...
                 (sol_pathAccMILP.Ffast+sol_pathAccMILP.Fslow);
epsilonPathMILP = (max(0,t_AvgPathMILP-Tmax)/Tmax).^2;
epsilonPathMILP = epsilonPathMILP(1:nOD);
fp_save = sprintf('output/nCar/%d/path_flows_pathAccMILP.mat',nCar);
[pathAccMILP(1,i_nCar,1),...
 pathAccMILP(1,i_nCar,2),...
 pathAccMILP(1,i_nCar,3)] = path_flows_Leo(Tmax,X_matrix,epsilonPathMILP',fp_save,D,B,G);

% Heatmap

% minTT
load(sprintf('output/nCar/%d/minTT.mat',nCar));
X_matrix = sol_mintt.X;
tm_TT = (2*(t'*X_matrix)'./sum(abs(D),1)');
epsilonTT = (max(0,60*(tm_TT-Tmax))).^2;
fp_load = sprintf('output/nCar/%d/path_flows_minTT.mat',nCar);
fp_save = sprintf('output/nCar/%d/AFI_heatmap_minTT.mat',nCar);
AFI_heatmap_sq(Tmax,fp_load,fp_save,epsilonTT,D,false)

% avgAcc
load(sprintf('output/nCar/%d/avgAcc.mat',nCar));
X_matrix = sol_avgAcc.X;
tm_avg = (t'*X_matrix)'./(sum(abs(D),1)'/2);
epsilonAvg = (max(0,60*(tm_avg-Tmax))).^2;
fp_load = sprintf('output/nCar/%d/path_flows_avgAcc.mat',nCar);
fp_save = sprintf('output/nCar/%d/AFI_heatmap_avgAcc.mat',nCar);
AFI_heatmap_sq(Tmax,fp_load,fp_save,epsilonAvg,D,false)

% pathAcc 
load(sprintf('output/nCar/%d/pathAcc.mat',nCar))
% X_matrix = sol_pathAcc.X;
Efast = max(0,60*(sol_pathAcc.tfast-Tmax)).^2;
Eslow = max(0,60*(sol_pathAcc.tslow-Tmax)).^2;
epsilonPath = (sol_pathAcc.Ffast.*Efast + ...
               sol_pathAcc.Fslow.*Eslow)/sum(alpha);
fp_load = sprintf('output/nCar/%d/path_flows_pathAcc.mat',nCar);
fp_save = sprintf('output/nCar/%d/AFI_heatmap_pathAcc.mat',nCar);
AFI_heatmap_sq(Tmax,fp_load,fp_save,epsilonPath',D,true)

% MILP
load(sprintf('output/nCar/%d/pathAccMILP.mat',nCar))
Efast = max(0,60*(sol_pathAccMILP.tfast-Tmax)).^2;
Eslow = max(0,60*(sol_pathAccMILP.tslow-Tmax)).^2;
epsilonPathMILP = (sol_pathAccMILP.Ffast.*Efast + ...
                   sol_pathAccMILP.Fslow.*Eslow)/sum(alpha);
fp_load = sprintf('output/nCar/%d/path_flows_pathAccMILP.mat',nCar);
fp_save = sprintf('output/nCar/%d/AFI_heatmap_pathAccMILP.mat',nCar);
AFI_heatmap_sq(Tmax,fp_load,fp_save,epsilonPathMILP',D,true)

% % This is for plotting in python
% MILP = sol_pathAccMILP.epsilon;
% str_save_afi = 'output/afi_MILP.mat';
% save(str_save_afi, "MILP")
end

str_save = sprintf('output/nCar/TT_AFI_%d.mat',nCar);
save(str_save,'minTT','avgAcc','pathAcc','pathAccMILP');



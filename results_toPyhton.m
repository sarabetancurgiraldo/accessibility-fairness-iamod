clear; close all; clc;

load("model/data_g.mat");
load("model/data_shortPaths.mat");

load('output/minTT.mat');
load('output/avgAcc.mat');
load('output/pathAcc.mat');

load("output/dest_deficit.mat");

Tmax = 20/60;
Nmin = 35;


AFIminTT = load('output/AFI_heatmap_minTT.mat');
AFIavgAcc = load('output/AFI_heatmap_avgAcc.mat');
AFIpathAcc = load('output/AFI_heatmap_pathAcc.mat');


fp_load = sprintf('output/AFI_heatmap_pathAccMILP.mat');
AFIdestDef = load(fp_load);
fp_load = sprintf('output/pathAccMILP.mat');
load(fp_load);


X_mintt = sol_mintt.X;
X_maxAcc = sol_avgAcc.X;
X_pathAcc = sol_pathAcc.X;
X_pathAccMILP = sol_pathAccMILP.X;

path_minTT = AFIminTT.AFI;
path_maxAcc = AFIavgAcc.AFI;
path_pathAcc = AFIpathAcc.AFI;
path_pathAccMILP = AFIdestDef.AFI;

avg_minTT = AFIminTT.AFI_epsilons;
avg_maxAcc = AFIavgAcc.AFI_epsilons;
avg_pathAcc = AFIpathAcc.AFI_epsilons;
avg_pathAccMILP = AFIdestDef.AFI_epsilons;

edgesMatlab = G.Edges.EndNodes;
times = G.Edges.Weight;
demand = sum(abs(D))/2;

%% Ur in matlab for python

% ur_tt_avg = population_region.*(R_selector*avg_minTT)/sum(population_region);
% ur_avgAcc_avg = population_region.*(R_selector*avg_maxAcc)/sum(population_region);
% ur_pathAcc_avg = population_region.*(R_selector*avg_pathAcc)/sum(population_region);
% 
% ur_tt_path = population_region.*(R_selector*path_minTT)/sum(population_region);
% ur_avgAcc_path = population_region.*(R_selector*path_maxAcc)/sum(population_region);
% ur_pathAcc_path = population_region.*(R_selector*path_pathAcc)/sum(population_region);


ur_tt_avg = R_selector*avg_minTT;
ur_avgAcc_avg = R_selector*avg_maxAcc;
ur_pathAcc_avg = R_selector*avg_pathAcc;

ur_tt_path = R_selector*path_minTT;
ur_avgAcc_path = R_selector*path_maxAcc;
ur_pathAcc_path = R_selector*path_pathAcc;


% ur_avgAcc_avg_dest = population_region.*(dest_def_OD_AvgAcc)/sum(population_region);
% % ur_pathAcc_avg_dest = population_region.*(dest_def_OD_pathAcc)/sum(population_region);
% % ur_dest_def_avg_dest = population_region.*(sol_pathAccMILP.epsilon)/sum(population_region);
% 
% % ur_avgAcc_path_dest = population_region.*(dest_def_path_AvgAcc)/sum(population_region);
% ur_pathAcc_path_dest = population_region.*(dest_def_path_pathAcc)/sum(population_region);
% ur_dest_def_path_dest = population_region.*(sol_pathAccMILP.epsilon)/sum(population_region);

ur_avgAcc_avg_dest = dest_def_OD_AvgAcc;
ur_pathAcc_path_dest = dest_def_path_pathAcc;
ur_dest_def_path_dest = sol_pathAccMILP.epsilon;

str_save = sprintf('output/matlab_python_T%d.mat',Tmax*60);
save(str_save, "X_mintt", "X_maxAcc", "X_pathAcc", "X_pathAccMILP", ...
               "path_minTT", "path_maxAcc", "path_pathAcc", "path_pathAccMILP", ...
               "avg_minTT", "avg_maxAcc", "avg_pathAcc", "avg_pathAccMILP", ...
               "ur_tt_avg", "ur_avgAcc_avg", "ur_pathAcc_avg", ...
               "ur_tt_path", "ur_avgAcc_path", "ur_pathAcc_path", ...
               "ur_avgAcc_avg_dest","ur_pathAcc_path_dest", ...
               "ur_dest_def_path_dest", ...
               "edgesMatlab", "demand", "pc_unique") 


clear; close all; clc;

load("model/data_g.mat");
load("model/data_shortPaths.mat");

load('output/avgAcc_sustReg.mat');

Tmax = 20/60;

AFIavgAccS = load('output/AFI_heatmap_avgAcc_sustReg.mat');

X_maxAccS = sol_avgAccS.X;

path_maxAccS = AFIavgAccS.AFI;

avg_maxAccS = AFIavgAccS.AFI_epsilons;

edgesMatlab = G.Edges.EndNodes;
times = G.Edges.Weight;
demand = sum(abs(D))/2;

%% Ur in matlab for python

ur_avgAccS_avg = R_selector*avg_maxAccS;

ur_avgAccS_path = R_selector*path_maxAccS;

str_save = sprintf('output/matlab_python_T%d_S.mat',Tmax*60);
save(str_save, "X_maxAccS", "path_maxAccS", "avg_maxAccS", ...
               "ur_avgAccS_avg", "ur_avgAccS_path", ...
               "edgesMatlab", "demand", "pc_unique") 


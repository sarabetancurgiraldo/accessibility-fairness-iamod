close all; clear; clc;

pwd
addpath(genpath("/home/mech001/20222295/Sara/YALMIP-master"));
fclose('all');

load('model/data_g.mat');
load("model/data_shortPaths.mat");
% load('output/TT_AFI.mat','minTT','pathAcc')

%% Variables

nCar = 4e3; 
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

sustReg     = zeros(1,1,3);

%% optimization

% avgAcc SustReg
str_save = 'output/avgAcc_sustReg.mat';
maxAverageAcc_sustReg(nCar,Tmax,str_save,G,B,D,nArcs);

%% path flow allocation

% avgAcc SustReg
load('output/avgAcc_sustReg.mat');
X_matrix = sol_avgAccS.X;
X_matrix = X_matrix(:,1:nOD);
epsilonAvgS = sol_avgAccS.epsilon;
epsilonAvgS = epsilonAvgS(1:nOD);
fp_save = 'output/path_flows_avgAcc_sustReg.mat';
[sustReg(1,1,1), ...
 sustReg(1,1,2), ...
 sustReg(1,1,3)] = path_flows_Leo(Tmax, X_matrix, epsilonAvgS, fp_save,D,B,G);

%% Heatmap

% avgAcc SustReg
load('output/avgAcc_sustReg.mat');
X_matrix = sol_avgAcc.X;
tm_avg = (t'*X_matrix)'./(sum(abs(D),1)'/2);
epsilonAvgS = (max(0,60*(tm_avg-Tmax))).^2;
fp_load = 'output/path_flows_avgAcc_sustReg.mat';
fp_save = 'output/AFI_heatmap_avgAcc_sustReg.mat';
AFI_heatmap_sq(Tmax,fp_load,fp_save,epsilonAvgS,D,false)

save('output/TT_AFI_S.mat','sustReg');


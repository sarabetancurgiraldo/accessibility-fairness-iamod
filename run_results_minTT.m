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

minTT = zeros(1,nC,3);

for i_nCar = 1:nC
nCar = nCarRange(i_nCar); 

% optimizations

% minTT
str_save = sprintf('output/nCar/%d/minTT_reg.mat',nCar);
minTravelTime_reg(nCar,G,B,nArcs,D,str_save);
% path flow allocation

% minTT
% load(sprintf('output/nCar/%d/minTT_reg.mat',nCar));
% X_matrix = sol_mintt.X;
% X_matrix = X_matrix(:,1:nOD);
% epsilonTT = (max(0,(2*(t'*X_matrix)'./sum(abs(D),1)')-Tmax)/Tmax).^2;
% fp_save = sprintf('output/nCar/%d/path_flows_minTT_reg.mat',nCar);
% [minTT(1,i_nCar,1), ...
%  minTT(1,i_nCar,2), ...
%  minTT(1,i_nCar,3)] = path_flows_Leo(Tmax,X_matrix,epsilonTT,fp_save,D,B,G);
% 
% % Heatmap
% 
% % minTT
% load(sprintf('output/nCar/%d/minTT_reg.mat',nCar));
% X_matrix = sol_mintt.X;
% tm_TT = (2*(t'*X_matrix)'./sum(abs(D),1)');
% epsilonTT = (max(0,60*(tm_TT-Tmax))).^2;
% fp_load = sprintf('output/nCar/%d/path_flows_minTT_reg.mat',nCar);
% fp_save = sprintf('output/nCar/%d/AFI_heatmap_minTT_reg.mat',nCar);
% AFI_heatmap_sq(Tmax,fp_load,fp_save,epsilonTT,D,false)

end

str_save = sprintf('output/nCar/TT_AFI_%d_minTT.mat',nCar);
save(str_save,'minTT');



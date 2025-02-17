clear; close all; clc;

load("model/data_g.mat");
load("model/data_shortPaths.mat");

% nCarRange       = [1e3 3e3 4e3]; 
nCarRange       = [ 3e3 ]; 

Tmax = 20/60;
Nmin = 35;

edgesMatlab = G.Edges.EndNodes;
times = G.Edges.Weight;
demand = sum(abs(D))/2;

nC = length(nCarRange);

for i_nCar = 1:nC
nCar = nCarRange(i_nCar);

fp_load = sprintf('output/nCar/%d/AFI_heatmap_MIQP.mat',nCar);
AFIdestDef = load(fp_load);
fp_load = sprintf('output/nCar/%d/MIQP.mat',nCar);
load(fp_load);

X_MIQP = sol_MIQP.X;

path_MIQP = AFIdestDef.AFI;

avg_MIQP = AFIdestDef.AFI_epsilons;

%% Ur in matlab for python

% ur_dest_def_path_dest = sol_MIQP.epsilon;
b_path = zeros(nOD,1); b_path(find(~path_MIQP)) = 1; 
ur_dest_def_path_dest = max(0,((Nmin-R_selector*b_path)/Nmin).^2);
MIQPobj_N = population_region'*ur_dest_def_path_dest/sum(population_region);%/Nmin;


% minTT DestDeficit
load(sprintf('output/nCar/%d/AFI_heatmap_minTT.mat',nCar));
b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
dest_def_OD_minTT = max(0,((Nmin-R_selector*b_OD)/Nmin).^2);
b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
dest_def_path_minTT = max(0,((Nmin-R_selector*b_path)/Nmin).^2);

% avg-Acc DestDeficit
load(sprintf('output/nCar/%d/AFI_heatmap_avgAcc.mat',nCar));
b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
dest_def_OD_avgAcc = max(0,((Nmin-R_selector*b_OD)/Nmin).^2);
b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
dest_def_path_avgAcc = max(0,((Nmin-R_selector*b_path)/Nmin).^2);

% avg-Acc-reg DestDeficit
load(sprintf('output/nCar/%d/AFI_heatmap_avgAcc_reg.mat',nCar));
b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
dest_def_OD_avgAcc_r = max(0,((Nmin-R_selector*b_OD)/Nmin).^2);
b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
dest_def_path_avgAcc_r = max(0,((Nmin-R_selector*b_path)/Nmin).^2);

% Path-Acc DestDeficit
load(sprintf('output/nCar/%d/AFI_heatmap_pathAcc.mat',nCar));
b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
dest_def_OD_pathAcc = max(0,((Nmin-R_selector*b_OD)/Nmin).^2);
b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
dest_def_path_pathAcc = max(0,((Nmin-R_selector*b_path)/Nmin).^2);


str_save = sprintf('output/nCar/%d/matlab_python_T%d_MIQP.mat',nCar,Tmax*60);
save(str_save, "X_MIQP", "path_MIQP", "avg_MIQP", ...
               "dest_def_OD_minTT","dest_def_path_minTT", ...
               "dest_def_OD_avgAcc","dest_def_path_avgAcc", ...
               "dest_def_OD_avgAcc_r","dest_def_path_avgAcc_r", ...
               "dest_def_OD_pathAcc","dest_def_path_pathAcc", ...
               "ur_dest_def_path_dest", "pc_unique","population_region") 


Jacc_OD_minTT = population_region'*dest_def_OD_minTT/sum(population_region);
Jacc_path_minTT = population_region'*dest_def_path_minTT/sum(population_region);
Jacc_OD_AvgAcc = population_region'*dest_def_OD_avgAcc/sum(population_region);
Jacc_path_AvgAcc = population_region'*dest_def_path_avgAcc/sum(population_region);
Jacc_OD_AvgAcc_r = population_region'*dest_def_OD_avgAcc_r/sum(population_region);
Jacc_path_AvgAcc_r = population_region'*dest_def_path_avgAcc_r/sum(population_region);
Jacc_OD_pathAcc = population_region'*dest_def_OD_pathAcc/sum(population_region);
Jacc_path_pathAcc = population_region'*dest_def_path_pathAcc/sum(population_region);



end

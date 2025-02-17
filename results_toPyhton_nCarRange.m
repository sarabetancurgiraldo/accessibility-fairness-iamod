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

% min TT DestDeficit
load(sprintf('output/nCar/%d/AFI_heatmap_minTT.mat',nCar));
b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
dest_def_OD_minTT = max(0,(Nmin-R_selector*b_OD)/Nmin);
deltaN_OD_minTT = population_region'*dest_def_OD_minTT/sum(population_region);%/Nmin;
b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
dest_def_path_minTT = max(0,(Nmin-R_selector*b_path)/Nmin);
deltaN_path_minTT = population_region'*dest_def_path_minTT/sum(population_region);%/Nmin;

save_str = sprintf("output/nCar/%d/dest_deficit_minTT.mat",nCar);
save(save_str, "dest_def_OD_minTT","dest_def_path_minTT")


load(sprintf('output/nCar/%d/minTT.mat',nCar));
load(sprintf('output/nCar/%d/avgAcc.mat',nCar));
load(sprintf('output/nCar/%d/pathAcc.mat',nCar));
load(sprintf("output/nCar/%d/dest_deficit.mat",nCar));


AFIminTT = load(sprintf('output/nCar/%d/AFI_heatmap_minTT.mat',nCar));
AFIavgAcc = load(sprintf('output/nCar/%d/AFI_heatmap_avgAcc.mat',nCar));
AFIpathAcc = load(sprintf('output/nCar/%d/AFI_heatmap_pathAcc.mat',nCar));


fp_load = sprintf('output/nCar/%d/AFI_heatmap_pathAccMILP.mat',nCar);
AFIdestDef = load(fp_load);
fp_load = sprintf('output/nCar/%d/pathAccMILP.mat',nCar);
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
%%
% avgAcc reg
if nCar == 3000
load(sprintf('output/nCar/%d/avgAcc_reg.mat',nCar));
AFIavgAcc_reg = load(sprintf('output/nCar/%d/AFI_heatmap_avgAcc_reg.mat',nCar));
X_maxAcc_reg = sol_avgAcc.X;
path_maxAcc_reg = AFIavgAcc_reg.AFI;
avg_maxAcc_reg = AFIavgAcc_reg.AFI_epsilons;
ur_avgAcc_avg_reg = R_selector*avg_maxAcc_reg;
ur_avgAcc_path_reg = R_selector*path_maxAcc_reg;
end
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

ur_minTT_avg_dest = dest_def_OD_minTT;
ur_minTT_path_dest = dest_def_path_minTT;
ur_avgAcc_avg_dest = dest_def_OD_AvgAcc;
ur_avgAcc_path_dest = dest_def_path_AvgAcc;
ur_pathAcc_avg_dest = dest_def_OD_pathAcc;
ur_pathAcc_path_dest = dest_def_path_pathAcc;
ur_dest_def_path_dest = sol_pathAccMILP.epsilon;

str_save = sprintf('output/nCar/%d/matlab_python_T%d.mat',nCar,Tmax*60);
save(str_save, "X_mintt", "X_maxAcc", "X_pathAcc", "X_pathAccMILP", "X_maxAcc_reg",...
               "path_minTT", "path_maxAcc", "path_pathAcc", "path_pathAccMILP","path_maxAcc_reg", ...
               "avg_minTT", "avg_maxAcc", "avg_pathAcc", "avg_pathAccMILP","avg_maxAcc_reg", ...
               "ur_tt_avg", "ur_avgAcc_avg", "ur_pathAcc_avg","ur_avgAcc_avg_reg", ...
               "ur_tt_path", "ur_avgAcc_path", "ur_pathAcc_path","ur_avgAcc_path_reg", ...
               "ur_minTT_avg_dest","ur_minTT_path_dest", ...
               "ur_avgAcc_avg_dest","ur_avgAcc_path_dest", ...
               "ur_pathAcc_avg_dest","ur_pathAcc_path_dest", ...
               "ur_dest_def_path_dest", ...
               "edgesMatlab", "demand", "pc_unique") 
end

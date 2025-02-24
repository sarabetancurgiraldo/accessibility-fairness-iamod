close all; clear; clc;

load('model/data_g.mat');
load("model/data_shortPaths.mat");




Tmax            = 20/60; 
% nCarRange       = [1e3 3e3 4e3]; 
nCarRange       = [3e3]; 
maxY            = 4000;
file_typ        = 'pdf';
% file_typ        = "png";
alpha           = sum(abs(D),1)/2;
Nmin            = 35;


% nOD             = 5;
% D               = D(:,1:nOD);
% Xfast           = Xfast(:,1:nOD);
% Xslow           = Xslow(:,1:nOD);
% R_selector      = R_selector(:,1:nOD);
% alpha           = alpha(:,1:nOD);

nC = length(nCarRange);

for i_nCar = 1:nC
nCar = nCarRange(i_nCar);

load('output/nCar/TT_AFI_3000_minTT.mat');

%%

% Fix this for deficit for minTT_reg
% % Avg-Acc DestDeficit
% load(sprintf('output/nCar/%d/AFI_heatmap_avgAcc.mat',nCar));
% b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
% dest_def_OD_AvgAcc = max(0,(Nmin-R_selector*b_OD)/Nmin);
% deltaN_OD_AvgAcc = population_region'*dest_def_OD_AvgAcc/sum(population_region);%/Nmin;
% b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
% dest_def_path_AvgAcc = max(0,(Nmin-R_selector*b_path)/Nmin);
% deltaN_path_AvgAcc = population_region'*dest_def_path_AvgAcc/sum(population_region);%/Nmin;
% 
% save_str = sprintf("output/nCar/%d/dest_deficit.mat",nCar);
% save(save_str, ...
%      "dest_def_OD_AvgAcc","dest_def_path_AvgAcc", ...
%      "dest_def_OD_pathAcc","dest_def_path_pathAcc")

% MinTT
Tavg = minTT(1,i_nCar,1);
% minTT, OD-based (average accessibility)
fp_load = sprintf('output/nCar/%d/minTT_reg.mat',nCar);
load(fp_load)
X = sol_mintt.X;
fp_save = sprintf('output/nCar/%d/plot/modal_share_OD_minTT_reg.mat',nCar);
fp_save_fig = sprintf('output/nCar/%d/figures/user/modal_share_OD_minTT_reg.%s',nCar,file_typ);
% metric1 = "Acc,Comm";
metric1 = "CommSuff";
obj_minTT_OD = minTT(1,i_nCar,3);
obj1 = sprintf("%0.4f",obj_minTT_OD);
l = leg(metric1,obj1,"min^2",0,0);
plot_modal_share_legend_user(Tmax,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
% minTT, path-based (path accessibility)
% fp_load = sprintf('output/nCar/%d/path_flows_minTT.mat',nCar);
% fp_save = sprintf('output/nCar/%d/plot/modal_share_path_minTT.mat',nCar);
% fp_save_fig = sprintf('output/nCar/%d/figures/user/modal_share_path_minTT.%s',nCar,file_typ);
% % metric1 = "Acc,Trip";
% metric1 = "TripSuff";
% obj_minTT_path = minTT(1,i_nCar,2);
% obj1 = sprintf("%0.4f",obj_minTT_path);
% l = leg(metric1,obj1,"min^2",0,0);
% plot_modal_share_legend_user(Tmax,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
%                         D,maxY,l);

%{
%% Modal share Diff

T_max = Tmax;

% % OD-based (Average)
minTT = load(sprintf('output/nCar/%d/plot/modal_share_OD_minTT.mat',nCar));
AvgAcc = load(sprintf('output/nCar/%d/plot/modal_share_OD_avgAcc.mat',nCar));
pathAcc = load(sprintf('output/nCar/%d/plot/modal_share_OD_pathAcc.mat',nCar));
destAcc = load(sprintf('output/nCar/%d/plot/modal_share_OD_destAcc.mat',nCar));

% AvgAcc vs minTT
fp_save = sprintf('output/nCar/%d/figures/user/modal_share_dif_OD_AvgTT.%s',nCar,file_typ);
plot_modal_share_dif_user(T_max, AvgAcc, minTT, fp_save, 'Commute')

% AvgAcc vs PathAcc 
fp_save = sprintf('output/nCar/%d/figures/user/modal_share_dif_OD_AvgPath.%s',nCar,file_typ);
plot_modal_share_dif_user(T_max, pathAcc, AvgAcc, fp_save, 'Commute')

% AvgAcc vs Dest
fp_save = sprintf('output/nCar/%d/figures/user/modal_share_dif_OD_AvgDest.%s',nCar,file_typ);
plot_modal_share_dif_user(T_max, destAcc, AvgAcc, fp_save, 'Commute')

% PathAcc vs Dest 
fp_save = sprintf('output/nCar/%d/figures/user/modal_share_dif_OD_PathDest.%s',nCar,file_typ);
plot_modal_share_dif_user(T_max, destAcc, pathAcc, fp_save, 'Commute')


% Path-based
minTT = load(sprintf('output/nCar/%d/plot/modal_share_path_minTT.mat',nCar));
AvgAcc = load(sprintf('output/nCar/%d/plot/modal_share_path_avgAcc.mat',nCar));
pathAcc = load(sprintf('output/nCar/%d/plot/modal_share_path_pathAcc.mat',nCar));
destAcc = load(sprintf('output/nCar/%d/plot/modal_share_path_destAcc.mat',nCar));

% AvgAcc vs minTT
fp_save = sprintf('output/nCar/%d/figures/user/modal_share_dif_path_AvgTT.%s',nCar,file_typ);
plot_modal_share_dif_user(T_max, AvgAcc, minTT, fp_save, 'Trip')

% AvgAcc vs PathAcc
fp_save = sprintf('output/nCar/%d/figures/user/modal_share_dif_path_AvgPath.%s',nCar,file_typ);
plot_modal_share_dif_user(T_max, pathAcc, AvgAcc, fp_save, 'Trip')

% AvgAcc vs dest
fp_save = sprintf('output/nCar/%d/figures/user/modal_share_dif_path_AvgDest.%s',nCar,file_typ);
plot_modal_share_dif_user(T_max, destAcc, AvgAcc, fp_save, 'Trip')

% PathAcc vs dest 
fp_save = sprintf('output/nCar/%d/figures/user/modal_share_dif_path_PathDest.%s',nCar,file_typ);
plot_modal_share_dif_user(T_max, destAcc, pathAcc, fp_save, 'Trip')
%}
end

function l = leg(m1,o1,u1,multi_obj,star_opt,m2,o2,u2)
l1 = ["$J_{\mathrm{",m1,"}}$"];
l2 = ["$",o1,"\ \mathrm{",u1,"}$"];
% l2 = ["$",o1,"$",u1];
if multi_obj
    l3 = ["$J_{\mathrm{",m2,"}}$"];
    l4 = ["$",o2,"\ \mathrm{",u2,"}$"];
%     l4 = ["$",o2,"$",u2];
end
if star_opt == 1
    l1 = ["$J_{\mathrm{",m1,"}}^{\star}$"];
elseif star_opt == 2
    l3 = ["$J_{\mathrm{",m2,"}}^{\star}$"];
end
l = {strjoin(l1),strjoin(l2)};
if multi_obj
    l = {strjoin(l1),strjoin(l2),strjoin(l3),strjoin(l4)};
end
end

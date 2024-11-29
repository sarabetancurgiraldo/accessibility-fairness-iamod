close all; clear; clc;

load('model/data_g.mat');
load("model/data_shortPaths.mat");

load('output/TT_AFI.mat');

Tmax            = 20/60; 
nCar            = 4e3;
maxY            = 2300;
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

%%
% Avg-Acc DestDeficit
load("output/AFI_heatmap_avgAcc.mat");
b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
dest_def_OD_AvgAcc = max(0,(Nmin-R_selector*b_OD)/Nmin);
deltaN_OD_AvgAcc = population_region'*dest_def_OD_AvgAcc/sum(population_region);%/Nmin;
b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
dest_def_path_AvgAcc = max(0,(Nmin-R_selector*b_path)/Nmin);
deltaN_path_AvgAcc = population_region'*dest_def_path_AvgAcc/sum(population_region);%/Nmin;

% Path-Acc DestDeficit
load("output/AFI_heatmap_pathAcc.mat");
b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
dest_def_OD_pathAcc = max(0,(Nmin-R_selector*b_OD)/Nmin);
deltaN_OD_pAcc = population_region'*dest_def_OD_pathAcc/sum(population_region);%/Nmin;
b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
dest_def_path_pathAcc = max(0,(Nmin-R_selector*b_path)/Nmin);
deltaN_path_pAcc = population_region'*dest_def_path_pathAcc/sum(population_region);%/Nmin;

save("output/dest_deficit.mat", ...
     "dest_def_OD_AvgAcc","dest_def_path_AvgAcc", ...
     "dest_def_OD_pathAcc","dest_def_path_pathAcc")

% MinTT
Tavg = minTT(1,1,1);
% minTT, OD-based (average accessibility)
fp_load = 'output/minTT.mat';
load(fp_load)
X = sol_mintt.X;
fp_save = 'output/plot/modal_share_OD_minTT.mat';
fp_save_fig = sprintf('output/figures/user/modal_share_OD_minTT.%s',file_typ);
metric1 = "Acc,Comm";
obj_minTT_OD = minTT(1,1,3);
obj1 = sprintf("%0.4f",obj_minTT_OD);
l = leg(metric1,obj1,"min^2",0,0);
plot_modal_share_legend_user(Tmax,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
% minTT, path-based (path accessibility)
fp_load = 'output/path_flows_minTT.mat';
fp_save = 'output/plot/modal_share_path_minTT.mat';
fp_save_fig = sprintf('output/figures/user/modal_share_path_minTT.%s',file_typ);
metric1 = "Acc,Trip";
obj_minTT_path = minTT(1,1,2);
obj1 = sprintf("%0.4f",obj_minTT_path);
l = leg(metric1,obj1,"min^2",0,0);
plot_modal_share_legend_user(Tmax,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);


%% AvgAcc
Tavg = avgAcc(1,1,1); 
% AvgAcc OD-metric 
fp_load = 'output/avgAcc.mat';
load(fp_load);
X = sol_avgAcc.X;
fp_save = 'output/plot/modal_share_OD_avgAcc.mat';
fp_save_fig = sprintf('output/figures/user/modal_share_OD_avgAcc.%s',file_typ);
metric1 = "Acc,Comm";
obj_avgAcc_OD = avgAcc(1,1,3);
obj1 = sprintf("%0.4f",obj_avgAcc_OD);
l = leg(metric1,obj1,"min^2",0,1);
plot_modal_share_legend_user(Tmax,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
fp_save = 'output/plot/modal_share_OD_dest_avgAcc.mat';
fp_save_fig = sprintf('output/figures/user/modal_share_OD_dest_avgAcc.%s',file_typ);
metric2 = "Acc,Dest";
obj2 = sprintf("%0.4f",deltaN_OD_AvgAcc);
l = leg(metric1,obj1,"min^2",1,1,metric2,obj2,"N\ dest");
plot_modal_share_legend_user(Tmax,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);

% avgAcc path-metric 
fp_load = 'output/path_flows_avgAcc.mat';
fp_save = 'output/plot/modal_share_path_avgAcc.mat';
fp_save_fig = sprintf('output/figures/user/modal_share_path_avgAcc.%s',file_typ);
metric1 = "Acc,Trip";
obj_avgAcc_path = avgAcc(1,1,2);
obj1 = sprintf("%0.4f",obj_avgAcc_path);
l = leg(metric1,obj1,"min^2",0,0);
plot_modal_share_legend_user(Tmax,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);
fp_save = 'output/plot/modal_share_path_dest_avgAcc.mat';
fp_save_fig = sprintf('output/figures/user/modal_share_path_dest_avgAcc.%s',file_typ);
metric2 = "Acc,Dest";
obj2 = sprintf("%0.4f",deltaN_path_AvgAcc);
l = leg(metric1,obj1,"min^2",1,0,metric2,obj2,"N\ dest");
plot_modal_share_legend_user(Tmax,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);

%% pathAcc 
Tavg = pathAcc(1,1,1); 
% pathAcc OD-metric 
fp_load = 'output/pathAcc.mat';
load(fp_load)
X = sol_pathAcc.X;
fp_save = 'output/plot/modal_share_OD_pathAcc.mat';
fp_save_fig = sprintf('output/figures/user/modal_share_OD_pathAcc.%s',file_typ);
metric1 = "Acc,Comm";
obj_pathAcc_OD = pathAcc(1,1,3);
obj1 = sprintf("%0.4f",obj_pathAcc_OD);
l = leg(metric1,obj1,"min^2",0,0);
plot_modal_share_legend_user(Tmax,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
fp_save = 'output/plot/modal_share_OD_dest_pathAcc.mat';
fp_save_fig = sprintf('output/figures/user/modal_share_OD_dest_pathAcc.%s',file_typ);
metric2 = "Acc,Dest";
obj2 = sprintf("%0.4f",deltaN_OD_pAcc);
l = leg(metric1,obj1,"min^2",1,0,metric2,obj2,"N\ dest");
plot_modal_share_legend_user(Tmax,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);

% pathAcc path-metric
fp_load = 'output/path_flows_pathAcc.mat';
fp_save = 'output/plot/modal_share_path_pathAcc.mat';
fp_save_fig = sprintf('output/figures/user/modal_share_path_pathAcc.%s',file_typ);
metric1 = "Acc,Trip";
obj_pathAcc_path = pathAcc(1,1,2);
obj1 = sprintf("%0.4f",obj_pathAcc_path);
l = leg(metric1,obj1,"min^2",0,1);
plot_modal_share_legend_user(Tmax,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);
fp_save = 'output/plot/modal_share_path_dest_pathAcc.mat';
fp_save_fig = sprintf('output/figures/user/modal_share_path_dest_pathAcc.%s',file_typ);
metric2 = "Acc,Dest";
obj2 = sprintf("%0.4f",deltaN_path_pAcc);
l = leg(metric1,obj1,"min^2",1,1,metric2,obj2,"N\ dest");
plot_modal_share_legend_user(Tmax,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);

% MILP

fp_load = 'output/pathAccMILP.mat';
load(fp_load)
load('output/AFI_heatmap_pathAccMILP.mat');
eps = (sol_pathAccMILP.epsilon)/Nmin;
MILPobj_N = population_region'*eps/sum(population_region);%/Nmin;
MILPobj_OD_t = pathAccMILP(1,1,3);
% MILPobj_OD_t = population_region'*(R_selector*(AFI_epsilons.*alpha')./(R_selector*alpha'))/sum(population_region);
MILPobj_path_t = pathAccMILP(1,1,2);
% MILPobj_path_t = population_region'*(R_selector*(AFI.*alpha')./(R_selector*alpha'))/sum(population_region);

Tavg = pathAccMILP(1,1,1); 
% pathAccMILP OD-based (average accessibility)
X = sol_pathAccMILP.X;
fp_save = 'output/plot/modal_share_OD_destAcc.mat';
fp_save_fig = sprintf('output/figures/user/modal_share_OD_destAcc.%s',file_typ);
metric1 = "Acc,Comm";
obj1 = sprintf("%0.4f",MILPobj_OD_t);
metric2 = "Acc,Dest";
obj2 = sprintf("%0.4f",MILPobj_N);
l = leg(metric1,obj1,"min^2",1,0,metric2,obj2,"N\ dest");
plot_modal_share_legend_user(Tmax,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
% pathAccMILP path-based (path accessibility)
fp_load = 'output/path_flows_pathAccMILP.mat';
fp_save = 'output/plot/modal_share_path_destAcc.mat';
fp_save_fig = sprintf('output/figures/user/modal_share_path_destAcc.%s',file_typ);
metric1 = "Acc,Trip";
obj1 = sprintf("%0.4f",MILPobj_path_t);
l = leg(metric1,obj1,"min^2",1,2,metric2,obj2,"N\ dest");
plot_modal_share_legend_user(Tmax,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);

%% Modal share Diff

T_max = Tmax;

% % OD-based (Average)
minTT = load('output/plot/modal_share_OD_minTT.mat');
AvgAcc = load('output/plot/modal_share_OD_avgAcc.mat');
pathAcc = load('output/plot/modal_share_OD_pathAcc.mat');
destAcc = load('output/plot/modal_share_OD_destAcc.mat');

% AvgAcc vs minTT
fp_save = sprintf('output/figures/user/modal_share_dif_OD_AvgTT.%s',file_typ);
plot_modal_share_dif_user(T_max, AvgAcc, minTT, fp_save, 'Average')

% AvgAcc vs PathAcc 
fp_save = sprintf('output/figures/user/modal_share_dif_OD_AvgPath.%s',file_typ);
plot_modal_share_dif_user(T_max, pathAcc, AvgAcc, fp_save, 'Average')

% AvgAcc vs Dest
fp_save = sprintf('output/figures/user/modal_share_dif_OD_AvgDest.%s',file_typ);
plot_modal_share_dif_user(T_max, destAcc, AvgAcc, fp_save, 'Average')

% PathAcc vs Dest 
fp_save = sprintf('output/figures/user/modal_share_dif_OD_PathDest.%s',file_typ);
plot_modal_share_dif_user(T_max, destAcc, pathAcc, fp_save, 'Average')


% Path-based
minTT = load('output/plot/modal_share_path_minTT.mat');
AvgAcc = load('output/plot/modal_share_path_avgAcc.mat');
pathAcc = load('output/plot/modal_share_path_pathAcc.mat');
destAcc = load('output/plot/modal_share_path_destAcc.mat');

% AvgAcc vs minTT
fp_save = sprintf('output/figures/user/modal_share_dif_path_AvgTT.%s',file_typ);
plot_modal_share_dif_user(T_max, AvgAcc, minTT, fp_save, 'Path')

% AvgAcc vs PathAcc
fp_save = sprintf('output/figures/user/modal_share_dif_path_AvgPath.%s',file_typ);
plot_modal_share_dif_user(T_max, pathAcc, AvgAcc, fp_save, 'Path')

% AvgAcc vs dest
fp_save = sprintf('output/figures/user/modal_share_dif_path_AvgDest.%s',file_typ);
plot_modal_share_dif_user(T_max, destAcc, AvgAcc, fp_save, 'Path')

% PathAcc vs dest 
fp_save = sprintf('output/figures/user/modal_share_dif_path_PathDest.%s',file_typ);
plot_modal_share_dif_user(T_max, destAcc, pathAcc, fp_save, 'Path')


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

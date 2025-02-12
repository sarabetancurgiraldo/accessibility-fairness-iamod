close all; clear; clc;

load('model/data_g.mat');
load("model/data_shortPaths.mat");




Tmax            = 20/60; 
% nCarRange       = [1e3 3e3 4e3]; 
nCarRange       = [3e3]; 
maxY            = 3500;
file_typ        = 'pdf';
% file_typ        = "png";
alpha           = sum(abs(D),1)/2;
Nmin            = 35;

nC = length(nCarRange);

for i_nCar = 1:nC
nCar = nCarRange(i_nCar);

load(sprintf('output/nCar/TT_AFI_MIQP_%d.mat',nCar));
load('output/nCar/TT_AFI_4000.mat');

% Path-Acc DestDeficit
load(sprintf('output/nCar/%d/AFI_heatmap_pathAcc.mat',nCar));
b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
dest_def_path_pathAcc = max(0,(Nmin-R_selector*b_path).^2/Nmin);
deltaN_path_pAcc = population_region'*dest_def_path_pathAcc/sum(population_region);%/Nmin;

Tavg = pathAcc(1,2,1); 
fp_load = sprintf('output/nCar/%d/pathAcc.mat',nCar);
load(fp_load)
X = sol_pathAcc.X;
fp_load = sprintf('output/nCar/%d/path_flows_pathAcc.mat',nCar);
metric1 = "TripSuff";
obj_pathAcc_path = pathAcc(1,2,2);
obj1 = sprintf("%0.4f",obj_pathAcc_path);
fp_save = sprintf('output/nCar/%d/plot/modal_share_path_dest_MIQP_pathAcc.mat',nCar);
fp_save_fig = sprintf('output/nCar/%d/figures/user/modal_share_path_dest_MIQP_pathAcc.%s',nCar,file_typ);
% metric2 = "Acc,Dest";
metric2 = "AccSuff";
obj2 = sprintf("%0.4f",deltaN_path_pAcc);
l = leg(metric1,obj1,"min^2",1,1,metric2,obj2,"");%"N\ dest");
plot_modal_share_legend_user(Tmax,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);

% MIQP

fp_load = sprintf('output/nCar/%d/MIQP.mat',nCar);
load(fp_load)
load(sprintf('output/nCar/%d/AFI_heatmap_MIQP.mat',nCar));
eps = (sol_MIQP.epsilon)/Nmin;
MIQPobj_N = population_region'*eps/sum(population_region);%/Nmin;
MIQPobj_OD_t = MIQP(1,i_nCar,3);
MIQPobj_path_t = MIQP(1,i_nCar,2);

Tavg = MIQP(1,i_nCar,1); 
% MIQP OD-based (average accessibility)
X = sol_MIQP.X;
fp_save = sprintf('output/nCar/%d/plot/modal_share_OD_destAccMIQP.mat',nCar);
fp_save_fig = sprintf('output/nCar/%d/figures/user/modal_share_OD_destAccMIQP.%s',nCar,file_typ);
% metric1 = "Acc,Comm";
metric1 = "CommSuff";
obj1 = sprintf("%0.4f",MIQPobj_OD_t);
% metric2 = "Acc,Dest";
metric2 = "AccSuff";
obj2 = sprintf("%0.4f",MIQPobj_N);
l = leg(metric1,obj1,"min^2",1,0,metric2,obj2,"");%"N\ dest");
plot_modal_share_legend_user(Tmax,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
% pathAccMILP path-based (path accessibility)
fp_load = sprintf('output/nCar/%d/path_flows_MIQP.mat',nCar);
fp_save = sprintf('output/nCar/%d/plot/modal_share_path_destAccMIQP.mat',nCar);
fp_save_fig = sprintf('output/nCar/%d/figures/user/modal_share_path_destAccMIQP.%s',nCar,file_typ);
% metric1 = "Acc,Trip";
metric1 = "TripSuff";
obj1 = sprintf("%0.4f",MIQPobj_path_t);
l = leg(metric1,obj1,"min^2",1,2,metric2,obj2,"");%"N\ dest");
plot_modal_share_legend_user(Tmax,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);

%% Modal share Diff

T_max = Tmax;

% % OD-based (Average)
% AvgAcc = load(sprintf('output/nCar/%d/plot/modal_share_OD_avgAcc.mat',nCar));
% pathAcc = load(sprintf('output/nCar/%d/plot/modal_share_OD_pathAcc.mat',nCar));
% destAccMIQP = load(sprintf('output/nCar/%d/plot/modal_share_OD_destAccMIQP.mat',nCar));

% AvgAcc vs MIQP
% fp_save = sprintf('output/nCar/%d/figures/user/modal_share_dif_OD_AvgDestMIQP.%s',nCar,file_typ);
% plot_modal_share_dif_user(T_max, destAccMIQP, AvgAcc, fp_save, 'Commute')

% PathAcc vs MIQP 
% fp_save = sprintf('output/nCar/%d/figures/user/modal_share_dif_OD_PathDestMIQP.%s',nCar,file_typ);
% plot_modal_share_dif_user(T_max, destAccMIQP, pathAcc, fp_save, 'Commute')


% Path-based
% AvgAcc = load(sprintf('output/nCar/%d/plot/modal_share_path_avgAcc.mat',nCar));
pathAcc = load(sprintf('output/nCar/%d/plot/modal_share_path_dest_MIQP_pathAcc.mat',nCar));
destAccMIQP = load(sprintf('output/nCar/%d/plot/modal_share_path_destAccMIQP.mat',nCar));

% AvgAcc vs dest
% fp_save = sprintf('output/nCar/%d/figures/user/modal_share_dif_path_AvgDestMIQP.%s',nCar,file_typ);
% plot_modal_share_dif_user(T_max, destAccMIQP, AvgAcc, fp_save, 'Trip')

% PathAcc vs dest 
fp_save = sprintf('output/nCar/%d/figures/user/modal_share_dif_path_MIQP_PathDestMIQP.%s',nCar,file_typ);
plot_modal_share_dif_user(T_max, destAccMIQP, pathAcc, fp_save, 'Trip')

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

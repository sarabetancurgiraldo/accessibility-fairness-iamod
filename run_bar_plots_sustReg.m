close all; clear; clc;

load('model/data_g.mat');
load("model/data_shortPaths.mat");

load('output/TT_AFI_S.mat');

Tmax            = 20/60; 
nCar            = 4e3;
maxY            = 8000;
% file_typ        = 'pdf';
file_typ        = "png";
alpha           = sum(abs(D),1)/2;
Nmin            = 35;


% nOD             = 5;
% D               = D(:,1:nOD);
% Xfast           = Xfast(:,1:nOD);
% Xslow           = Xslow(:,1:nOD);
% R_selector      = R_selector(:,1:nOD);
% alpha           = alpha(:,1:nOD);

%% AvgAcc sustReg
Tavg = sustReg(1,1,1); 
% AvgAccS OD-metric 
fp_load = 'output/avgAcc_sustReg.mat';
load(fp_load);
X = sol_avgAccS.X;
fp_save = 'output/plot/modal_share_OD_avgAcc_sustReg.mat';
fp_save_fig = sprintf('output/figures/modal_share_OD_avgAcc_sustReg.%s',file_typ);
metric1 = "AccS,Avg";
obj_avgAccS_OD = sustReg(1,1,3);
obj1 = sprintf("%0.4f",obj_avgAccS_OD);
l = leg(metric1,obj1,"min^2",0,1);
plot_modal_share_legend(Tmax,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);

% avgAcc path-metric 
fp_load = 'output/path_flows_avgAcc_sustReg.mat';
fp_save = 'output/plot/modal_share_path_avgAcc_sustReg.mat';
fp_save_fig = sprintf('output/figures/modal_share_path_avgAcc_sustReg.%s',file_typ);
metric1 = "AccS,Path";
obj_avgAccS_path = sustReg(1,1,2);
obj1 = sprintf("%0.4f",obj_avgAccS_path);
l = leg(metric1,obj1,"min^2",0,0);
plot_modal_share_legend(Tmax,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);

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

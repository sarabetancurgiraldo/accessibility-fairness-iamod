close all; clear; clc;

load('model/data_g.mat');
load("model/data_shortPaths.mat");

load('output/J_.mat');
% load('output/J_AccSuff.mat');

maxY            = 8500;
file_typ        = 'pdf';
% file_typ        = "png";
alpha           = sum(abs(D),1)/2;


% nOD             = 5;
% D               = D(:,1:nOD);
% R_selector      = R_selector(:,1:nOD);
% alpha           = alpha(:,1:nOD);

nCarRange = [0 2e3 3e3 4e3 5e3];
TsuffRange = [15/60 20/60 25/60];
NsuffRange = [30 35 40];

nC = length(nCarRange);
Ts = length(TsuffRange);
Ns = length(NsuffRange);

for i_nCar = 1:nC
nCar = nCarRange(i_nCar); 

for i_Tsuff = 1:Ts
Tsuff = TsuffRange(i_Tsuff);

for i_Nsuff = 1:Ns
Nsuff = NsuffRange(i_Nsuff);

%%
if nCar ~= 0
% CommSuff DestDeficit
load(sprintf('output/nCar/%d/Tsuff/%d/AFI_heatmap_CommSuff.mat',nCar,Tsuff*60));
b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
dest_def_comm_CommSuff = max(0,Nsuff-R_selector*b_OD);
deltaN_comm_CommSuff = population_region'*dest_def_comm_CommSuff/sum(population_region)/Nsuff;
b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
dest_def_trip_CommSuff = max(0,Nsuff-R_selector*b_path);
deltaN_trip_CommSuff = population_region'*dest_def_trip_CommSuff/sum(population_region)/Nsuff;

% TripSuff DestDeficit
load(sprintf('output/nCar/%d/Tsuff/%d/AFI_heatmap_TripSuff.mat',nCar,Tsuff*60));
b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
dest_def_comm_TripSuff = max(0,Nsuff-R_selector*b_OD);
deltaN_comm_TripSuff = population_region'*dest_def_comm_TripSuff/sum(population_region)/Nsuff;
b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
dest_def_trip_TripSuff = max(0,Nsuff-R_selector*b_path);
deltaN_trip_TripSuff = population_region'*dest_def_trip_TripSuff/sum(population_region)/Nsuff;

save(sprintf("output/plot/Nsuff/%d/nCar/%d/Tsuff/%d/dest_deficit.mat",Nsuff,nCar,Tsuff*60), ...
             "dest_def_comm_CommSuff","dest_def_trip_CommSuff", ...
             "dest_def_comm_TripSuff","dest_def_trip_TripSuff", ...
             "deltaN_comm_CommSuff","deltaN_trip_CommSuff", ...
             "deltaN_comm_TripSuff","deltaN_trip_TripSuff")
end
end

%% UtilitarianEfficiency
Tavg = UtilEff(i_Tsuff,i_nCar,1);
% UtilitarianEfficiency, Commute-based
fp_load = sprintf('output/nCar/%d/Tsuff/%d/UtilEff.mat',nCar,Tsuff*60);
load(fp_load);
X = sol_utilEff.X;
fp_save = sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_comm_UtilEff.mat',nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/nCar/%d/Tsuff/%d/modal_share_comm_UtilEff.%s',nCar,Tsuff*60,file_typ);
metric1 = "CommSuff";
obj_UtilEff_comm = UtilEff(i_Tsuff,i_nCar,3);
obj1 = sprintf("%0.4f",obj_UtilEff_comm);
l = leg(metric1,obj1,"min^2",0,0);
plot_modal_share_legend_user(Tsuff,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
% UtilitarianEfficiency, Trip-based
fp_load = sprintf('output/nCar/%d/Tsuff/%d/path_flows_UtilEff.mat',nCar,Tsuff*60);
fp_save = sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_trip_UtilEff.mat',nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/nCar/%d/Tsuff/%d/modal_share_trip_UtilEff.%s',nCar,Tsuff*60,file_typ);
metric1 = "TripSuff";
obj_UtilEff_trip = UtilEff(i_Tsuff,i_nCar,2);
obj1 = sprintf("%0.4f",obj_UtilEff_trip);
l = leg(metric1,obj1,"min^2",0,0);
plot_modal_share_legend_user(Tsuff,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);

if nCar ~= 0

%% CommuteSufficiency
Tavg = CommSuff(i_Tsuff,i_nCar,1); 
% CommuteSufficiency Commute-metric 
fp_load = sprintf('output/nCar/%d/Tsuff/%d/CommSuff.mat',nCar,Tsuff*60);
load(fp_load);
X = sol_comSuff.X;
fp_save = sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_comm_CommSuff.mat',nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/nCar/%d/Tsuff/%d/modal_share_comm_CommSuff.%s',nCar,Tsuff*60,file_typ);
metric1 = "CommSuff";
obj_commSuff_comm = CommSuff(i_Tsuff,i_nCar,3);
obj1 = sprintf("%0.4f",obj_commSuff_comm);
l = leg(metric1,obj1,"min^2",0,1);
plot_modal_share_legend_user(Tsuff,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
for i_Nsuff = 1:Ns
Nsuff = NsuffRange(i_Nsuff);
load(sprintf("output/plot/Nsuff/%d/dest_deficit.mat",Nsuff))
fp_save = sprintf('output/plot/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_comm_dest_CommSuff.mat',Nsuff,nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_comm_dest_CommSuff.%s',Nsuff,nCar,Tsuff*60,file_typ);
metric2 = "AccSuff";
obj2 = sprintf("%0.4f",deltaN_comm_CommSuff);
l = leg(metric1,obj1,"min^2",1,1,metric2,obj2,"N\ dest");
plot_modal_share_legend_user(Tsuff,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
end

% CommuteSufficiency trip-metric 
fp_load = sprintf('output/nCar/%d/Tsuff/%d/path_flows_CommSuff.mat',nCar,Tsuff*60);
fp_save = sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_trip_CommSuff.mat',nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/nCar/%d/Tsuff/%d/modal_share_trip_CommSuff.%s',nCar,Tsuff*60,file_typ);
metric1 = "TripSuff";
obj_commSuff_trip = CommSuff(i_Tsuff,i_nCar,2);
obj1 = sprintf("%0.4f",obj_commSuff_trip);
l = leg(metric1,obj1,"min^2",0,0);
plot_modal_share_legend_user(Tsuff,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);
for i_Nsuff = 1:Ns
Nsuff = NsuffRange(i_Nsuff);
load(sprintf("output/plot/Nsuff/%d/dest_deficit.mat",Nsuff))
fp_save = sprintf('output/plot/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_trip_dest_CommSuff.mat',Nsuff,nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_trip_dest_CommSuff.%s',Nsuff,nCar,Tsuff*60,file_typ);
metric2 = "AccSuff";
obj2 = sprintf("%0.4f",deltaN_trip_CommSuff);
l = leg(metric1,obj1,"min^2",1,0,metric2,obj2,"N\ dest");
plot_modal_share_legend_user(Tsuff,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
end

%% TripSufficiency 
Tavg = TripSuff(i_Tsuff,i_nCar,1); 
% TripSufficiency Commute-metric 
fp_load = sprintf('output/nCar/%d/Tsuff/%d/TripSuff.mat',nCar,Tsuff*60);
load(fp_load)
X = sol_Tripsuff.X;
fp_save = sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_comm_TripSuff.mat',nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/nCar/%d/Tsuff/%d/modal_share_comm_TripSuff.%s',nCar,Tsuff*60,file_typ);
metric1 = "CommSuff";
obj_TripSuff_comm = TripSuff(i_Tsuff,i_nCar,3);
obj1 = sprintf("%0.4f",obj_TripSuff_comm);
l = leg(metric1,obj1,"min^2",0,0);
plot_modal_share_legend_user(Tsuff,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
for i_Nsuff = 1:Ns
Nsuff = NsuffRange(i_Nsuff);
load(sprintf("output/plot/Nsuff/%d/dest_deficit.mat",Nsuff))
fp_save = sprintf('output/plot/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_comm_dest_TripSuff.mat',Nsuff,nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_comm_dest_TripSuff.%s',Nsuff,nCar,Tsuff*60,file_typ);
metric2 = "AccSuff";
obj2 = sprintf("%0.4f",deltaN_comm_TripSuff);
l = leg(metric1,obj1,"min^2",1,0,metric2,obj2,"N\ dest");
plot_modal_share_legend_user(Tsuff,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
end

% TripSufficiency trip-metric
fp_load = sprintf('output/nCar/%d/Tsuff/%d/path_flows_TripSuff.mat',nCar,Tsuff*60);
fp_save = sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_trip_TripSuff.mat',nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/nCar/%d/Tsuff/%d/modal_share_trip_TripSuff.%s',nCar,Tsuff*60,file_typ);
metric1 = "TripSuff";
obj_tripSuff_trip = TripSuff(i_Tsuff,i_nCar,2);
obj1 = sprintf("%0.4f",obj_tripSuff_trip);
l = leg(metric1,obj1,"min^2",0,1);
plot_modal_share_legend_user(Tsuff,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);
for i_Nsuff = 1:Ns
Nsuff = NsuffRange(i_Nsuff);
load(sprintf("output/plot/Nsuff/%d/dest_deficit.mat",Nsuff))
fp_save = sprintf('output/plot/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_trip_dest_TripSuff.mat',Nsuff,nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_trip_dest_TripSuff.%s',Nsuff,nCar,Tsuff*60,file_typ);
metric2 = "AccSuff";
obj2 = sprintf("%0.4f",deltaN_trip_TripSuff);
l = leg(metric1,obj1,"min^2",1,1,metric2,obj2,"N\ dest");
plot_modal_share_legend_user(Tsuff,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
end

%% AccessibilitySufficiency
for i_Nsuff = 1:Ns
Nsuff = NsuffRange(i_Nsuff);

load(sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/J.mat',Nsuff,nCar,Tsuff*60));
fp_load = sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/AccSuff.mat',Nsuff,nCar,Tsuff*60);
load(fp_load);
load(sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/AFI_heatmap_AccSuff.mat',Nsuff,nCar,Tsuff*60));
AccSuffObj_N = population_region'*sol_AccSuff.u_r/sum(population_region)/Nsuff;
AccSuffObj_comm_t = AccSuff{i_Nsuff,i_Tsuff,i_nCar,3};
AccSuffObj_trip_t = AccSuff{i_Nsuff,i_Tsuff,i_nCar,2};

Tavg = AccSuff{i_Nsuff,i_Tsuff,i_nCar,1}; 
% AccessibilitySufficiency commute-based 
X = sol_AccSuff.X;
fp_save = sprintf('output/plot/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_comm_AccSuff.mat',Nsuff,nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_comm_AccSuff.%s',Nsuff,nCar,Tsuff*60,file_typ);
metric1 = "CommSuff";
obj1 = sprintf("%0.4f",AccSuffObj_comm_t);
metric2 = "AccSuff";
obj2 = sprintf("%0.4f",AccSuffObj_N);
l = leg(metric1,obj1,"min^2",1,0,metric2,obj2,"N\ dest");
plot_modal_share_legend_user(Tsuff,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);
% AccessibilitySufficiency trip-based 
fp_load = sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/path_flows_AccSuff.mat',Nsuff,nCar,Tsuff*60);
fp_save = sprintf('output/plot/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_trip_AccSuff.mat',Nsuff,nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_trip_AccSuff.%s',Nsuff,nCar,Tsuff*60,file_typ);
metric1 = "TripSuff";
obj1 = sprintf("%0.4f",AccSuffObj_trip_t);
l = leg(metric1,obj1,"min^2",1,2,metric2,obj2,"N\ dest");
plot_modal_share_legend_user(Tsuff,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);
end
end
end
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

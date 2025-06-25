clear; close all; clc;

load("model/data_g.mat");
load("model/data_shortPaths.mat");

load('output/J_.mat');
% load('output/J_AccSuff.mat');

maxY = 4000;
Nsuff = 35;
nCar = 3e3;
Tsuff = 20/60;
file_typ = 'pdf';
alpha = sum(abs(D),1)/2;

%% Plots

% Values
% CommSuff DestDeficit
load(sprintf('output/nCar/%d/Tsuff/%d/AFI_heatmap_CommSuff.mat',nCar,Tsuff*60));
b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
dest_def_comm_CommSuff = max(0,((Nsuff-R_selector*b_OD)/Nsuff).^2);
deltaN_comm_CommSuff = population_region'*dest_def_comm_CommSuff/sum(population_region);
b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
dest_def_trip_CommSuff = max(0,((Nsuff-R_selector*b_path)/Nsuff).^2);
deltaN_trip_CommSuff = population_region'*dest_def_trip_CommSuff/sum(population_region);

% TripSuff DestDeficit
load(sprintf('output/nCar/%d/Tsuff/%d/AFI_heatmap_TripSuff.mat',nCar,Tsuff*60));
b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
dest_def_comm_TripSuff = max(0,((Nsuff-R_selector*b_OD)/Nsuff).^2);
deltaN_comm_TripSuff = population_region'*dest_def_comm_TripSuff/sum(population_region);
b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
dest_def_trip_TripSuff = max(0,((Nsuff-R_selector*b_path)/Nsuff).^2);
deltaN_trip_TripSuff = population_region'*dest_def_trip_TripSuff/sum(population_region);

%% UtilitarianEfficiency
Tavg = UtilEff(2,3,1);
% UtilitarianEfficiency, Commute-based
fp_load = sprintf('output/nCar/%d/Tsuff/%d/UtilEff.mat',nCar,Tsuff*60);
load(fp_load);
X = sol_utilEff.X;
fp_save = sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_comm_UtilEff.mat',nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/nominal/modal_share_comm_UtilEff.%s',file_typ);
metric1 = "CommSuff";
obj_UtilEff_comm = UtilEff(2,3,3);
obj1 = sprintf("%0.4f",obj_UtilEff_comm);
l = leg(metric1,obj1,"min^2",0,0);
plot_modal_share_legend_user(Tsuff,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);


%% CommuteSufficiency
Tavg = CommSuff(2,3,1); 
% CommuteSufficiency Commute-metric 
fp_load = sprintf('output/nCar/%d/Tsuff/%d/CommSuff.mat',nCar,Tsuff*60);
load(fp_load);
X = sol_comSuff.X;
fp_save = sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_comm_CommSuff.mat',nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/nominal/modal_share_comm_CommSuff.%s',file_typ);
metric1 = "CommSuff";
obj_commSuff_comm = CommSuff(2,3,3);
obj1 = sprintf("%0.4f",obj_commSuff_comm);
l = leg(metric1,obj1,"min^2",0,1);
plot_modal_share_legend_user(Tsuff,false,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);

% CommuteSufficiency trip-metric 
fp_load = sprintf('output/nCar/%d/Tsuff/%d/path_flows_CommSuff.mat',nCar,Tsuff*60);
fp_save = sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_trip_CommSuff.mat',nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/nominal/modal_share_trip_CommSuff.%s',file_typ);
metric1 = "TripSuff";
obj_commSuff_trip = CommSuff(2,3,2);
obj1 = sprintf("%0.4f",obj_commSuff_trip);
l = leg(metric1,obj1,"min^2",0,0);
plot_modal_share_legend_user(Tsuff,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);

%% TripSufficiency 
Tavg = TripSuff(2,3,1); 
fp_load = sprintf('output/nCar/%d/Tsuff/%d/TripSuff.mat',nCar,Tsuff*60);
load(fp_load)
X = sol_Tripsuff.X;

% TripSufficiency trip-metric
fp_load = sprintf('output/nCar/%d/Tsuff/%d/path_flows_TripSuff.mat',nCar,Tsuff*60);
fp_save = sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_trip_TripSuff.mat',nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/nominal/modal_share_trip_TripSuff.%s',file_typ);
metric1 = "TripSuff";
obj_tripSuff_trip = TripSuff(2,3,2);
obj1 = sprintf("%0.4f",obj_tripSuff_trip);
l = leg(metric1,obj1,"min^2",0,1);
plot_modal_share_legend_user(Tsuff,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);

fp_save = sprintf('output/plot/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_trip_dest_TripSuff.mat',Nsuff,nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/nominal/modal_share_trip_dest_TripSuff.%s',file_typ);
metric2 = "AccSuff";
obj2 = sprintf("%0.4f",deltaN_trip_TripSuff);
l = leg(metric1,obj1,"min^2",1,1,metric2,obj2,"");
plot_modal_share_legend_user(Tsuff,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l,X);

%% AccessibilitySufficiency
% load(sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/J.mat',Nsuff,nCar,Tsuff*60));
fp_load = sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/AccSuff.mat',Nsuff,nCar,Tsuff*60);
load(fp_load);
load(sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/AFI_heatmap_AccSuff.mat',Nsuff,nCar,Tsuff*60));
AccSuffObj_N = population_region'*sol_AccSuff.u_r/sum(population_region)/Nsuff;
AccSuffObj_comm_t = AccSuff{2,2,3,3};
AccSuffObj_trip_t = AccSuff{2,2,3,2};

Tavg = AccSuff{2,2,3,1}; 
X = sol_AccSuff.X;

% AccessibilitySufficiency trip-based 
fp_load = sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/path_flows_AccSuff.mat',Nsuff,nCar,Tsuff*60);
fp_save = sprintf('output/plot/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_trip_AccSuff.mat',Nsuff,nCar,Tsuff*60);
fp_save_fig = sprintf('output/figures/nominal/modal_share_trip_AccSuff.%s',file_typ);
metric1 = "TripSuff";
obj1 = sprintf("%0.4f",AccSuffObj_trip_t);
metric2 = "AccSuff";
obj2 = sprintf("%0.4f",AccSuffObj_N);
l = leg(metric1,obj1,"min^2",1,2,metric2,obj2,"");
plot_modal_share_legend_user(Tsuff,true,fp_load,fp_save,fp_save_fig,Tavg,G, ...
                        D,maxY,l);


%% Differences

% Commute metric
utilEff_comm = load(sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_comm_UtilEff.mat',nCar,Tsuff*60));
commSuff_comm = load(sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_comm_CommSuff.mat',nCar,Tsuff*60));

% Trip metric
commSuff_trip = load(sprintf('output/plot/nCar/%d/Tsuff/%d/modal_share_trip_CommSuff.mat',nCar,Tsuff*60));
tripSuff_trip = load(sprintf('output/plot/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_trip_dest_TripSuff.mat',Nsuff,nCar,Tsuff*60));
accSuff_trip = load(sprintf('output/plot/Nsuff/%d/nCar/%d/Tsuff/%d/modal_share_trip_AccSuff.mat',Nsuff,nCar,Tsuff*60));


% UtilEff vs CommSuff comm metric
fp_save = sprintf('output/figures/nominal/modal_share_dif_comm_UtilCommSuff.%s',file_typ);
plot_modal_share_dif_user(Tsuff, commSuff_comm, utilEff_comm, fp_save, 'Commute')


% TripSuff vs CommSuff trip metric
fp_save = sprintf('output/figures/nominal/modal_share_dif_trip_TripCommSuff.%s',file_typ);
plot_modal_share_dif_user(Tsuff, tripSuff_trip, commSuff_trip, fp_save, 'Trip')

% AccSuff vs TripCuff trip metric
fp_save = sprintf('output/figures/nominal/modal_share_dif_trip_AccTripSuff.%s',file_typ);
plot_modal_share_dif_user(Tsuff, accSuff_trip, tripSuff_trip, fp_save, 'Trip')

%% Ur in matlab for python

% Utilitarian Efficiency
load(sprintf('output/nCar/%d/Tsuff/%d/AFI_heatmap_UtilEff.mat',nCar,Tsuff*60));
% Commute metric
% Commute Insufficiency
commInsuff_comm_utilEff = R_selector * AFI_epsilons;
% % Accessibility Insufficiency
% b_comm = zeros(nOD,1); b_comm(find(~AFI_epsilons)) = 1; 
% accInsuff_comm_utilEff = max(0,((Nsuff-R_selector*b_comm)/Nsuff).^2);


% Commute Sufficiency
load(sprintf('output/nCar/%d/Tsuff/%d/AFI_heatmap_CommSuff.mat',nCar,Tsuff*60));
% Commute metric
% Commute Insufficiency
commInsuff_comm_commSuff = R_selector * AFI_epsilons;
% % Accessibility Insufficiency
% b_comm = zeros(nOD,1); b_comm(find(~AFI_epsilons)) = 1; 
% accInsuff_comm_commSuff = max(0,((Nsuff-R_selector*b_comm)/Nsuff).^2);
% Trip metric
% Trip Insufficiency
tripInsuff_trip_commSuff = R_selector * AFI;
% % Accessibility Insufficiency
% b_trip = zeros(nOD,1); b_trip(find(~AFI)) = 1; 
% accInsuff_trip_commSuff = max(0,((Nsuff-R_selector*b_trip)/Nsuff).^2);


% Trip Sufficiency
load(sprintf('output/nCar/%d/Tsuff/%d/AFI_heatmap_TripSuff.mat',nCar,Tsuff*60));
% Trip metric
% Trip Insufficiency
tripInsuff_trip_tripSuff = R_selector * AFI;
% Accessibility Insufficiency
b_trip = zeros(nOD,1); b_trip(find(~AFI)) = 1; 
accInsuff_trip_tripSuff = max(0,((Nsuff-R_selector*b_trip)/Nsuff).^2);


% Accessibility Sufficiency
% Trip metric
% Accessibility Insufficiency
load(sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/AFI_heatmap_AccSuff.mat',Nsuff,nCar,Tsuff*60));
b_trip = zeros(nOD,1); b_trip(find(~AFI)) = 1; 
accInsuff_accSuff = max(0,((Nsuff-R_selector*b_trip)/Nsuff).^2);

fp_load = sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/AccSuff.mat',Nsuff,nCar,Tsuff*60);
load(fp_load);

X_accSuff = sol_AccSuff.X;

edgesMatlab = G.Edges.EndNodes;
% times = G.Edges.Weight;
demand = sum(abs(D))/2;

str_save = 'output/data_accInsuff_TripAccSuff.mat';
save(str_save, "X_accSuff", "edgesMatlab", "demand", ...
               "commInsuff_comm_utilEff", "commInsuff_comm_commSuff", ...
               "tripInsuff_trip_commSuff", "tripInsuff_trip_tripSuff", ...
               "accInsuff_trip_tripSuff","accInsuff_accSuff", ...
               "pc_unique","population_region") 

%%



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




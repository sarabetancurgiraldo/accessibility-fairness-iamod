close all; clear; clc;

load("model\data_g.mat");
load("model\data_shortPaths.mat")

t               = G.Edges.Weight;
tfast           = t'*Xfast;

Tmax = 20/60;

od_acc = R_selector * (tfast<Tmax)';
od_region = sum(R_selector,2);

% n_regions = size(R_selector,1);
OD_framework = {};

for i = 1:4
    if i == 1
        load("output/AFI_heatmap_minTT.mat");
    elseif i == 2
        load("output/AFI_heatmap_avgAcc.mat");
    elseif i == 3
        load("output/AFI_heatmap_pathAcc.mat");
    elseif i == 4
        load("output/AFI_heatmap_pathAccMILP.mat");
    end

    b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
    OD_framework{1,i} = R_selector*b_OD;

end

dest_data = [pc_unique od_region od_acc ];

T = table(dest_data(:,1),dest_data(:,2),dest_data(:,3), ...
          OD_framework{1,1},OD_framework{1,2},OD_framework{1,3}, ...
          OD_framework{1,4},population_region, ...
          'VariableNames', ...
          ["PC","OD desired","OD shortPath", ...
          "OD minTT","OD commAcc","OD tripAcc","OD destAcc", ...
          "Population"]);







close all; clear; clc;

load("model\data_g.mat");
load("model\data_shortPaths.mat")

t               = G.Edges.Weight;
tfast           = t'*Xfast;

Tmax = 20/60;

od_acc = R_selector * (tfast<Tmax)';
od_region = sum(R_selector,2);

[pc_unique od_region od_acc]







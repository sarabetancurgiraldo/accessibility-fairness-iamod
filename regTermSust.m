% Calculate regularization term minTT vs AvgAcc
close all; clear; clc;

load('model/data_g.mat');
load("model/data_shortPaths.mat");
load('output/avgAcc_sustReg.mat');
load('output/minTT.mat');

%% Variables

X = sol_mintt.X;
% X = sol_avgAccS.X;

nCar = 4e3; 
Tmax = 20/60;

alpha           = sum(abs(D),1)/2;
t               = G.Edges.Weight;

speed_walk      = 5 / 3.6;
speed_bike      = 15 / 3.6;
speed_car       = 25 / 3.6;

arcsCar         = find(G.Edges.Type == 1);
arcsBike        = find(G.Edges.Type == 2);
arcsWalk        = find(G.Edges.Type == 3);
arcsPT          = find(G.Edges.Type == 4 | G.Edges.Type == 5);


dist_car        = sum(speed_car * (t(arcsCar)' * X(arcsCar,:))');
dist_bike       = sum(speed_bike * (t(arcsBike)' * X(arcsBike,:))');
dist_walk       = sum(speed_walk * (t(arcsWalk)' * X(arcsWalk,:))');
dist_pt         = sum(speed_car * (t(arcsPT)' * X(arcsPT,:))');

reg_dist        = dist_car + dist_bike/50 + dist_walk/25 + dist_pt/4


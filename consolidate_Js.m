%% Consolidate Js

close all; clear; clc;

load('model/data_g.mat');
load("model/data_shortPaths.mat");

J_02 = load('output/J_02.mat');
J_3 = load('output/J_3.mat');
J_45 = load('output/J_45.mat');

UtilEff = zeros(size(J_02.UtilEff));
CommSuff = zeros(size(J_02.CommSuff)); 
TripSuff = zeros(size(J_02.TripSuff)); 
AccSuff = zeros(size(J_02.AccSuff)); 


UtilEff(:,1:2,:) = J_02.UtilEff;
UtilEff(:,3,:) = J_3.UtilEff;
UtilEff(:,4:5,:) = J_45.UtilEff;


CommSuff(:,1:2,:) = J_02.CommSuff;
CommSuff(:,3,:) = J_3.CommSuff;
CommSuff(:,4:5,:) = J_45.CommSuff;


TripSuff(:,1:2,:) = J_02.TripSuff;
TripSuff(:,3,:) = J_3.TripSuff;
TripSuff(:,4:5,:) = J_45.TripSuff;


AccSuff(:,1:2,:) = J_02.AccSuff;
AccSuff(:,3,:) = J_3.AccSuff;
AccSuff(:,4:5,:) = J_45.AccSuff;


str_save = sprintf('output/J_.mat');
save(str_save,'UtilEff','CommSuff','TripSuff','AccSuff');







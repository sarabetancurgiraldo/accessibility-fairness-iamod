%% Consolidate Js

close all; clear; clc;

load('model/data_g.mat');
load("model/data_shortPaths.mat");

J_03 = load('output/J.mat');
J_45 = load('output/J_45.mat');

UtilEff = zeros(size(J_03.UtilEff));
CommSuff = zeros(size(J_03.CommSuff)); 
TripSuff = zeros(size(J_03.TripSuff)); 
AccSuff = zeros(size(J_03.AccSuff)); 


UtilEff(:,1:3,:) = J_03.UtilEff;
UtilEff(:,4:5,:) = J_45.UtilEff;


CommSuff(:,1:3,:) = J_03.CommSuff;
CommSuff(:,4:5,:) = J_45.CommSuff;


TripSuff(:,1:3,:) = J_03.TripSuff;
TripSuff(:,4:5,:) = J_45.TripSuff;


AccSuff(:,1:3,:) = J_03.AccSuff;
AccSuff(:,4:5,:) = J_45.AccSuff;


str_save = sprintf('output/J_.mat');
save(str_save,'UtilEff','CommSuff','TripSuff','AccSuff');







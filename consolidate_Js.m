%% Consolidate Js

close all; clear; clc;

load('model/data_g.mat');
load("model/data_shortPaths.mat");

J_02 = load('output/J_02.mat');
J_3 = load('output/J_3.mat');
J_45 = load('output/J_45.mat');


Ns = 3; % NsuffRange
UtilEff = zeros(size(J_02.UtilEff));
CommSuff = zeros(size(J_02.CommSuff)); 
TripSuff = zeros(size(J_02.TripSuff)); 
AccSuff = cell(Ns);%(Ts,nC,3);
% AccSuff = zeros(size(J_02.AccSuff)); 


UtilEff(:,1:2,:) = J_02.UtilEff(:,1:2,:);
UtilEff(:,3,:) = J_3.UtilEff(:,3,:);
UtilEff(:,4:5,:) = J_45.UtilEff(:,4:5,:);


CommSuff(:,1:2,:) = J_02.CommSuff(:,1:2,:);
CommSuff(:,3,:) = J_3.CommSuff(:,3,:);
CommSuff(:,4:5,:) = J_45.CommSuff(:,4:5,:);


TripSuff(:,1:2,:) = J_02.TripSuff(:,1:2,:);
TripSuff(:,3,:) = J_3.TripSuff(:,3,:);
TripSuff(:,4:5,:) = J_45.TripSuff(:,4:5,:);

% {i_Nsuff,i_Tsuff,i_nCar,i} (i=1: tt, i=2: pathSuff, i=3: commSuff)
for i_Ns = 1:3
    for i_Ts = 1:3
        for i = 1:3
            AccSuff{i_Ns, i_Ts, 1, i}   = J_02.AccSuff{i_Ns, i_Ts, 1, i};
            AccSuff{i_Ns, i_Ts, 2, i}   = J_02.AccSuff{i_Ns, i_Ts, 2, i};
            AccSuff{i_Ns, i_Ts, 3, i}   = J_3.AccSuff{i_Ns, i_Ts, 3, i};
            AccSuff{i_Ns, i_Ts, 4, i}   = J_45.AccSuff{i_Ns, i_Ts, 4, i};
            AccSuff{i_Ns, i_Ts, 5, i}   = J_45.AccSuff{i_Ns, i_Ts, 5, i};
        end
    end

end

str_save = sprintf('output/J_.mat');
save(str_save,'UtilEff','CommSuff','TripSuff','AccSuff');







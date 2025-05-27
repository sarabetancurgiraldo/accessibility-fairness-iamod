close all; clear; clc;

load('model/data_g.mat');
load("model/data_shortPaths.mat");
load('output/J.mat');
load('output/J_AccSuff.mat');

nCarRange = [0 2e3 3e3 4e3 5e3];
TsuffRange = [15/60 20/60 25/60];
NsuffRange = [30 35 40];

Ts = length(TsuffRange);
nC = length(nCarRange);
Ns = length(NsuffRange);

%{
%%
for i_nCar = 1:nC
for i_Tsuff = 1:Ts
for i_Nsuff = 1:Ns

nCar = nCarRange(i_nCar);
Tsuff = TsuffRange(i_Tsuff);
Nsuff = NsuffRange(i_Nsuff);

load(sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/J.mat',Nsuff,nCar,Tsuff*60));
fp_load = sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/AccSuff.mat',Nsuff,nCar,Tsuff*60);
load(fp_load);
load(sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/AFI_heatmap_AccSuff.mat',Nsuff,nCar,Tsuff*60));

AccSuffObj_N = population_region'*sol_AccSuff.u_r/sum(population_region)/Nsuff;

Tavg = AccSuff(1,1,1); 
TripSuff_ = AccSuff(1,1,2); 
CommSuff_ = AccSuff(1,1,3); 

sprintf(['nCar: %d \n' ...
        'Tsuff: %d \n' ...
        'Nsuff: %d \n'...
        'AccSuff: %.4f \n' ...
        'Travel time: %.4f \n' ...
        'CommSuff: %.4f \n' ...
        'TripSuff: %.4f'], nCar, Tsuff*60, Nsuff, AccSuffObj_N, Tavg, CommSuff_, TripSuff_)

% pause
end
end
end

%%

nCarRange = [2e3 3e3 4e3 5e3];

nC = length(nCarRange);

for i_nCar = 1:nC
for i_Tsuff = 1:Ts
for i_Nsuff = 1:Ns

nCar = nCarRange(i_nCar);
Tsuff = TsuffRange(i_Tsuff);
Nsuff = NsuffRange(i_Nsuff);

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


sprintf(['nCar: %d \n' ...
        'Tsuff: %d \n' ...
        'Nsuff: %d \n'...
        'AccSuff Comm Comm: %.4f \n' ...
        'AccSuff Comm Trip: %.4f \n' ...
        'AccSuff Trip Comm: %.4f \n' ...
        'AccSuff Trip Trip: %.4f'], ...
        nCar, Tsuff*60, Nsuff, ...
        deltaN_comm_CommSuff, deltaN_trip_CommSuff, deltaN_comm_TripSuff, deltaN_trip_TripSuff)


end
end
end

%%
load('output/J_2000.mat')

nCarRange = [2e3];
TsuffRange = [15/60 20/60 25/60];
NsuffRange = [30 35 40];


nC = length(nCarRange);

for i_nCar = 1:nC
for i_Tsuff = 1:Ts
for i_Nsuff = 1:Ns

nCar = nCarRange(i_nCar);
Tsuff = TsuffRange(i_Tsuff);
Nsuff = NsuffRange(i_Nsuff);


Tavg = AccSuff{i_Nsuff,i_Tsuff,i_nCar,1};
TripInsuff = AccSuff{i_Nsuff,i_Tsuff,i_nCar,2};
CommInsuff = AccSuff{i_Nsuff,i_Tsuff,i_nCar,3};

fp_load = sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/AccSuff.mat',Nsuff,nCar,Tsuff*60);
load(fp_load);

load(sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/AFI_heatmap_AccSuff.mat',Nsuff,nCar,Tsuff*60));
AccSuffObj_N = population_region'*sol_AccSuff.u_r/sum(population_region)/Nsuff;

sprintf(['nCar: %d \n' ...
        'Tsuff: %d \n' ...
        'Nsuff: %d \n'...
        'Travel time: %.4f \n' ...
        'CommSuff: %.4f \n' ...
        'TripSuff: %.4f \n' ...
        'AccSuff: %.4f'], ...
        nCar, Tsuff*60, Nsuff, Tavg, CommInsuff, TripInsuff, AccSuffObj_N)





end
end
end
%}
%% Create table
dim_values = Ts*nC*Ns;

% Matrix for each optimization objective 
    % UE: Utilitarian Efficiency
    % CS: Commute Sufficiency
    % TS: Trip Sufficiency
    % AS: Accessibility Sufficiency
% Columns values
% TT | CommInsuff | TripInsuff | AccInsuff (comm) | AccInsuff (trip)

UE = zeros(dim_values,5);
CS = zeros(dim_values,5);
TS = zeros(dim_values,5);
AS = zeros(dim_values,5);

for i_nCar = 1:nC
for i_Tsuff = 1:Ts
for i_Nsuff = 1:Ns

nCar = nCarRange(i_nCar);
Tsuff = TsuffRange(i_Tsuff);
Nsuff = NsuffRange(i_Nsuff);


if nCar == 0

    Tavg = UtilEff(i_Tsuff,i_nCar,1);
    UE(1:Ts*Ns,1) = Tavg * ones(Ts*Ns,1);
    CS(1:Ts*Ns,1) = Tavg * ones(Ts*Ns,1);
    TS(1:Ts*Ns,1) = Tavg * ones(Ts*Ns,1);
    AS(1:Ts*Ns,1) = Tavg * ones(Ts*Ns,1);

    CommInsuff = UtilEff(i_Tsuff,i_nCar,3);
    UE(1+(Ts*(i_Tsuff-1)):Ns+(Ts*(i_Tsuff-1)),2) = CommInsuff * ones(Ns,1);
    CS(1+(Ts*(i_Tsuff-1)):Ns+(Ts*(i_Tsuff-1)),2) = CommInsuff * ones(Ns,1);
    TS(1+(Ts*(i_Tsuff-1)):Ns+(Ts*(i_Tsuff-1)),2) = CommInsuff * ones(Ns,1);
    AS(1+(Ts*(i_Tsuff-1)):Ns+(Ts*(i_Tsuff-1)),2) = CommInsuff * ones(Ns,1);
    
    TripInsuff = UtilEff(i_Tsuff,i_nCar,2);
    UE(1+(Ts*(i_Tsuff-1)):Ns+(Ts*(i_Tsuff-1)),3) = TripInsuff * ones(Ns,1);
    CS(1+(Ts*(i_Tsuff-1)):Ns+(Ts*(i_Tsuff-1)),3) = TripInsuff * ones(Ns,1);
    TS(1+(Ts*(i_Tsuff-1)):Ns+(Ts*(i_Tsuff-1)),3) = TripInsuff * ones(Ns,1);
    AS(1+(Ts*(i_Tsuff-1)):Ns+(Ts*(i_Tsuff-1)),3) = TripInsuff * ones(Ns,1);
    
    load(sprintf('output/nCar/%d/Tsuff/%d/AFI_heatmap_UtilEff.mat',nCar,Tsuff*60));
    b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
    dest_def_comm_UtilEff = max(0,((Nsuff-R_selector*b_OD)/Nsuff).^2);
    deltaN_comm_UtilEff = population_region'*dest_def_comm_UtilEff/sum(population_region);%/Nsuff;
    b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
    dest_def_trip_UtilEff = max(0,((Nsuff-R_selector*b_path)/Nsuff).^2);
    deltaN_trip_UtilEff = population_region'*dest_def_trip_UtilEff/sum(population_region);%/Nsuff;
    
    UE(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),4) = deltaN_comm_UtilEff;
    CS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),4) = deltaN_comm_UtilEff;
    TS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),4) = deltaN_comm_UtilEff;
    AS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),4) = deltaN_comm_UtilEff;
    
    UE(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),5) = deltaN_trip_UtilEff;
    CS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),5) = deltaN_trip_UtilEff;
    TS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),5) = deltaN_trip_UtilEff;
%     AS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),5) = deltaN_trip_UtilEff;
    
else
    % TT
    Tavg = UtilEff(i_Tsuff,i_nCar,1);
    UE(1+((i_nCar-1)*Ts*Ns):Ts*Ns+((i_nCar-1)*Ts*Ns),1) = Tavg * ones(Ts*Ns,1);
    
    Tavg = CommSuff(i_Tsuff,i_nCar,1);
    CS(1+((i_nCar-1)*Ts*Ns):Ts*Ns+((i_nCar-1)*Ts*Ns),1) = Tavg * ones(Ts*Ns,1);
    
    Tavg = TripSuff(i_Tsuff,i_nCar,1);
    TS(1+((i_nCar-1)*Ts*Ns):Ts*Ns+((i_nCar-1)*Ts*Ns),1) = Tavg * ones(Ts*Ns,1);

    Tavg = AccSuff{i_Nsuff,i_Tsuff,i_nCar,1};
    AS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),1) = Tavg;
    
    % Commute Insufficiency
    CommInsuff = UtilEff(i_Tsuff,i_nCar,3);
    UE(1+((i_nCar-1)*Ts*Ns)+(Ts*(i_Tsuff-1)):Ns+((i_nCar-1)*Ts*Ns)+(Ts*(i_Tsuff-1)),2) = CommInsuff * ones(Ns,1);

    CommInsuff = CommSuff(i_Tsuff,i_nCar,3);
    CS(1+((i_nCar-1)*Ts*Ns)+(Ts*(i_Tsuff-1)):Ns+((i_nCar-1)*Ts*Ns)+(Ts*(i_Tsuff-1)),2) = CommInsuff * ones(Ns,1);

    CommInsuff = TripSuff(i_Tsuff,i_nCar,3);
    TS(1+((i_nCar-1)*Ts*Ns)+(Ts*(i_Tsuff-1)):Ns+((i_nCar-1)*Ts*Ns)+(Ts*(i_Tsuff-1)),2) = CommInsuff * ones(Ns,1);
    
    CommInsuff = AccSuff{i_Nsuff,i_Tsuff,i_nCar,3};
    AS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),2) = CommInsuff;
    
    % Trip Insufficiency
    TripInsuff = UtilEff(i_Tsuff,i_nCar,2);
    UE(1+((i_nCar-1)*Ts*Ns)+(Ts*(i_Tsuff-1)):Ns+((i_nCar-1)*Ts*Ns)+(Ts*(i_Tsuff-1)),3) = TripInsuff * ones(Ns,1);

    TripInsuff = CommSuff(i_Tsuff,i_nCar,2);
    CS(1+((i_nCar-1)*Ts*Ns)+(Ts*(i_Tsuff-1)):Ns+((i_nCar-1)*Ts*Ns)+(Ts*(i_Tsuff-1)),3) = TripInsuff * ones(Ns,1);

    TripInsuff = TripSuff(i_Tsuff,i_nCar,2);
    TS(1+((i_nCar-1)*Ts*Ns)+(Ts*(i_Tsuff-1)):Ns+((i_nCar-1)*Ts*Ns)+(Ts*(i_Tsuff-1)),3) = TripInsuff * ones(Ns,1);
    
    TripInsuff = AccSuff{i_Nsuff,i_Tsuff,i_nCar,2};
    AS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),3) = TripInsuff;

    % Accessibility Insufficiency
    % Utilitarian Efficiency
    load(sprintf('output/nCar/%d/Tsuff/%d/AFI_heatmap_UtilEff.mat',nCar,Tsuff*60));
    b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
    dest_def_comm_UtilEff = max(0,((Nsuff-R_selector*b_OD)/Nsuff).^2);
    deltaN_comm_UtilEff = population_region'*dest_def_comm_UtilEff/sum(population_region);%/Nsuff;
    b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
    dest_def_trip_UtilEff = max(0,((Nsuff-R_selector*b_path)/Nsuff).^2);
    deltaN_trip_UtilEff = population_region'*dest_def_trip_UtilEff/sum(population_region);%/Nsuff;

    UE(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),4) = deltaN_comm_UtilEff;
    UE(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),5) = deltaN_trip_UtilEff;

    % Commute Sufficiency
    load(sprintf('output/nCar/%d/Tsuff/%d/AFI_heatmap_CommSuff.mat',nCar,Tsuff*60));
    b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
    dest_def_comm_CommSuff = max(0,((Nsuff-R_selector*b_OD)/Nsuff).^2);
    deltaN_comm_CommSuff = population_region'*dest_def_comm_CommSuff/sum(population_region);%/Nsuff;
    b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
    dest_def_trip_CommSuff = max(0,((Nsuff-R_selector*b_path)/Nsuff).^2);
    deltaN_trip_CommSuff = population_region'*dest_def_trip_CommSuff/sum(population_region);%/Nsuff;
    
    CS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),4) = deltaN_comm_CommSuff;
    CS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),5) = deltaN_trip_CommSuff;

    % Trip Sufficiency
    load(sprintf('output/nCar/%d/Tsuff/%d/AFI_heatmap_TripSuff.mat',nCar,Tsuff*60));
    b_OD = zeros(nOD,1); b_OD(find(~AFI_epsilons)) = 1; 
    dest_def_comm_TripSuff = max(0,((Nsuff-R_selector*b_OD)/Nsuff).^2);
    deltaN_comm_TripSuff = population_region'*dest_def_comm_TripSuff/sum(population_region);%/Nsuff;
    b_path = zeros(nOD,1); b_path(find(~AFI)) = 1; 
    dest_def_trip_TripSuff = max(0,((Nsuff-R_selector*b_path)/Nsuff).^2);
    deltaN_trip_TripSuff = population_region'*dest_def_trip_TripSuff/sum(population_region);%/Nsuff;

    TS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),4) = deltaN_comm_TripSuff;
    TS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),5) = deltaN_trip_TripSuff;


    %Accessibility Sufficiency
    load(sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/AccSuff.mat',Nsuff,nCar,Tsuff*60));
    load(sprintf('output/Nsuff/%d/nCar/%d/Tsuff/%d/AFI_heatmap_AccSuff.mat',Nsuff,nCar,Tsuff*60));
    AccSuffObj_N = population_region'*sol_AccSuff.u_r/sum(population_region)/Nsuff;

    AS(1+((i_nCar-1)*Ts*Ns)+((i_Tsuff-1)*Ts)+(i_Nsuff-1),4) = AccSuffObj_N;

    
end
end
end
end



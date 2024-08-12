function plot_modal_share_legend(T_max,accType,fp_load,fp_save, ...
                                 fp_save_fig,Tavg,G,D,maxY,l,X)

%% Notes
% dif -> paths/OD pairs
% Tavg as vertical

%% Parameters
step_bins_min = 1; % min
save_fig = true;

%% Load data 
% Load model
% load('model/data_g.mat');
load(fp_load);

mask_car = find(G.Edges.Type==1);
bike_mask = find(G.Edges.Type==2);
walk_mask = find(G.Edges.Type==3 | G.Edges.Type==6);
pt_mask = find(G.Edges.Type==4);
wait_mask = find(G.Edges.Type==5);

%% Compute times per OD pair / path
if accType
    % Compute number of paths
    N_paths = 0;
    for idx_OD = 1:size(D,2)
        N_paths = N_paths + length(f_paths_cell{idx_OD,1});
    end    
    % Compute time for mode for each of pair
    T1_car = zeros(N_paths,1);
    T1_bike = zeros(N_paths,1);
    T1_walk = zeros(N_paths,1);
    T1_pt = zeros(N_paths,1);
    T1_wait = zeros(N_paths,1);    
    t = G.Edges.Weight';
    % Car
    t_aux_car = t;
    t_aux_car(find(G.Edges.Type~=1)) = 0;
    % t_aux_car = zeros(nArcs, 1); t_aux_car(mask_car) = t(mask_car)
    % Bike
    t_aux_bike = t;
    t_aux_bike(find(G.Edges.Type~=2)) = 0;
    % t_aux_bike = zeros(nArcs, 1); t_aux_bike(bike_mask) = t(bike_mask)
    % Walk
    t_aux_walk = t;
    t_aux_walk(find(~(G.Edges.Type==3 | G.Edges.Type==6))) = 0;
    % t_aux_walk = zeros(nArcs, 1); t_aux_walk(walk_mask) = t(walk_mask)
    % PT
    t_aux_pt = t;
    t_aux_pt(find(G.Edges.Type~=4)) = 0;
    % t_aux_pt = zeros(nArcs, 1); t_aux_pt(pt_mask) = t(pt_mask)
    % Wait
    t_aux_wait = t;
    t_aux_wait(find(G.Edges.Type~=5)) = 0;
    % t_aux_wait = zeros(nArcs, 1); t_aux_wait(wait_mask) = t(wait_mask)
    T_ = zeros(N_paths,1);
    idx_path = 0;
    for idx_OD = 1:size(D,2)
        for idx_path_OD = 1:length(f_paths_cell{idx_OD,1})
            idx_path = idx_path+1;       
            % Car 
            T1_car(idx_path) = t_aux_car(arc_mask_paths_cell{idx_OD})*...
                X_bin_paths_cell{idx_OD,1}(:,idx_path_OD)*...
                f_paths_cell{idx_OD,1}(idx_path_OD);
            % Bike
            T1_bike(idx_path) = t_aux_bike(arc_mask_paths_cell{idx_OD})*...
                X_bin_paths_cell{idx_OD,1}(:,idx_path_OD)*...
                f_paths_cell{idx_OD,1}(idx_path_OD);
            % Walk
            T1_walk(idx_path) = t_aux_walk(arc_mask_paths_cell{idx_OD})*...
                X_bin_paths_cell{idx_OD,1}(:,idx_path_OD)*...
                f_paths_cell{idx_OD,1}(idx_path_OD);
            % PT
            T1_pt(idx_path) = t_aux_pt(arc_mask_paths_cell{idx_OD})*...
                X_bin_paths_cell{idx_OD,1}(:,idx_path_OD)*...
                f_paths_cell{idx_OD,1}(idx_path_OD);
            % Wait
            T1_wait(idx_path) = t_aux_wait(arc_mask_paths_cell{idx_OD})*...
                X_bin_paths_cell{idx_OD,1}(:,idx_path_OD)*...
                f_paths_cell{idx_OD,1}(idx_path_OD);
            % Tbar
            T_(idx_path) = t_paths_cell{idx_OD,1}(idx_path_OD);
        end
    end
else 
    % Compute average per mode
    t = G.Edges.Weight';
    T_ = t*X./(sum(abs(D),1)/2)-1e-10; %time of od-pairs
    % Compute time for mode for each od pair
    T1_car = t(mask_car) * X(mask_car,:);
    T1_bike = t(bike_mask) * X(bike_mask,:);
    T1_walk = t(walk_mask) * X(walk_mask,:);
    T1_pt = t(pt_mask) * X(pt_mask,:);
    T1_wait = t(wait_mask) * X(wait_mask,:);
end

% Order times
step_bins = step_bins_min/60;
max_t_round = 60;%ceil(max(T_)*60/2); %Max t in minutes (múltiplo de 2)
edges = 0:step_bins:max_t_round*step_bins;

% od-times divided into edges (every 2 mins) 
% N(i) = k if edges(k) < T_(i) <= edges(k+1)
N = discretize(T_,edges,'IncludedEdge','right'); 

T =[];
for i = 1:length(edges)-1
    ind = find(N==i);
    car_bar_t = sum(T1_car(ind));
    bike_bar_t = sum(T1_bike(ind));
    walk_bar_T = sum(T1_walk(ind));
    pt_bar_t = sum(T1_pt(ind));
    wait_bar_t = sum(T1_wait(ind));
    T = [T; car_bar_t, bike_bar_t, walk_bar_T, pt_bar_t, wait_bar_t]; 
    x_vals(i) = ((edges(i)+edges(i+1))/2)*60;
end

save(fp_save,'T');

figure('Position',3*[0 0 192 (2/3)*144],'visible','off');
hold on;
grid on;
box on;
set(gca,'ticklabelinterpreter','Latex','fontsize',20)
set(groot,'defaulttextinterpreter','latex')
set(groot,'defaultaxesticklabelinterpreter','latex')
set(groot,'defaultlegendinterpreter','latex')
set(gca,'ticklabelinterpreter','Latex','fontsize',20)
bar(x_vals, T,'stacked')
ylim([0 maxY])
xlim([0 edges(end)*60])
set(gca,'ticklabelinterpreter','Latex','fontsize',14)
color.green = [0 158 115]/255;
x_Tmax = T_max*60;
xline(x_Tmax,'Linewidth',3,'color',color.green) %5.4
xticks(0:5:60); 

x_Tavg = Tavg;
xline(x_Tavg,'--','Linewidth',3,'Color','k'); %5.4

legend('Car','Bike','Walk','PT','Waiting PT','$T_\mathrm{max}$','$T_\mathrm{avg}$','fontsize',12);
t = annotation("textbox");
t.FontSize = 14;
t.Interpreter = 'latex';
t.Position = [0.4 0.91 0.01 0.01];
t.FitBoxToText ="on";
t.LineStyle = "none";
t.String = l;

if accType
    xlabel('Path Travel Time $[\mathrm{min}]$','fontsize',14);
else  
    xlabel('Average Travel Time $[\mathrm{min}]$','fontsize',14);
end
ylabel('Time-based Modal Share $[\mathrm{h}]$','fontsize',14)


if save_fig
    exportgraphics(gcf,fp_save_fig);
end


end

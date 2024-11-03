function plot_modal_share_dif(T_max, T1, T2, fp_save, lgd)
% Ti - time-based modal share. T1 - positive, T2 - negative

step_bins = 1; % min
% save_fig = false;
save_fig = true;

% load('model/data_g.mat','D','B','G','nArcs','nNodes','full_demand');

% Order times
%n_bins/xlim = integer; -> xlim = n_bins/integer = 4/6 -> integer = round(n_bins/(4/6))
% edges = 0:step_bins/60:round(40/step_bins)*step_bins/60; %linspace(0,nbins/round(nbins/(4/6)),nbins)

% max_t_round = size(TT.T,1)*step_bins;
nbins = 60*step_bins;
edges = 0:step_bins:nbins;

for i = 1:length(edges)-1
    x_vals(i) = ((edges(i)+edges(i+1))/2);
end


figure('Position',3*[0 0 192 (2/3)*144],'visible','off');
hold on;
grid on;
box on;
set(gca,'ticklabelinterpreter','Latex','fontsize',20)
set(groot,'defaulttextinterpreter','latex')
set(groot,'defaultaxesticklabelinterpreter','latex')
set(groot,'defaultlegendinterpreter','latex')
set(gca,'ticklabelinterpreter','Latex','fontsize',20)
bar(x_vals, T1.T-T2.T,'stacked')

ylim([-1050 2000])

xlim([0 nbins])
% xlim([0 max_t_round])
set(gca,'ticklabelinterpreter','Latex','fontsize',14)
color.green = [0 158 115]/255;
% x_Tmax = 60*T_max/step_bins + 0.5;
x_Tmax = T_max*60;
xline(x_Tmax,'Linewidth',3,'color',color.green) %5.4
% legend('Car','Bike','Walk','PT','Waiting PT','$T_\mathrm{max}$','fontsize',12); 
xlabel([lgd,' Travel Time $[\mathrm{min}]$'],'fontsize',14);
ylabel('Time-based Modal Share $[\mathrm{h}]$','fontsize',14);
xticks(0:5:60); 
% xticks(edges); 
% vector = [];
% for ii=1:nbins
%     vector(ii) = ii;
% end
% if ~acc_flag
%     xticklabels([]);
% else 
%     xticks(vector)
%     xticklabels(round(60*edges(2:end),2)-step_bins/2)
% end

% lgd = legend('Car','Bike','Walk','PT','Waiting PT','$T_\mathrm{max}$');
% lgd.FontSize = 12;

if save_fig
    exportgraphics(gcf,fp_save);
end




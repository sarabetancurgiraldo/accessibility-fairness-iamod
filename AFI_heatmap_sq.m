function AFI_heatmap_sq(T_max,fp_load,fp_save,epsilon,D,path)
    %% Load model data
%     load('model/data_g.mat','D'); % Data G
    alpha = sum(abs(D),1)'/2;
    
    %% Output varibles
    AFI_ = zeros(size(D,2),1);

    %% Load flow data - path accessibility
    
    % Load flows from path flows leo

    load(fp_load,'t_paths_cell','f_paths_cell');

    for idx_OD = 1:size(D,2)
        aux = (max(0,60*(t_paths_cell{idx_OD,1}-T_max))).^2;
        AFI_(idx_OD) = f_paths_cell{idx_OD,1}'*aux/sum(f_paths_cell{idx_OD,1});
    end
    AFI = alpha.*AFI_/sum(alpha);
%     AFI = AFI_;
    %% Load flow data - OD accessibility
    if path 
        AFI_epsilons = epsilon;
    else
        AFI_epsilons = alpha.*epsilon/sum(alpha);
    end
%     AFI_epsilons = epsilon;
    %% Save data
    save(fp_save,'AFI','AFI_epsilons');

end

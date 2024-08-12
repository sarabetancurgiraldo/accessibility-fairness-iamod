function [TT,AFI,AFImean] = path_flows_Leo(T_max,X_matrix,epsilon,fp_save, ...
                                           D,B,G)
%% Parameters
save_flows = true;

%% Load data
% Data from network flow model
% load('model/data_g.mat','D','B','G'); 

% AFImean = 60*epsilon'*sum(abs(D),1)'/(sum(sum(sum(abs(D)))));
AFImean = 60*epsilon'*sum(abs(D),1)'/(sum(abs(D),'all'));

%% Compute path flows
maxAcc = zeros(size(X_matrix,2),1);
minAcc = zeros(size(X_matrix,2),1);
minVarAcc = zeros(size(X_matrix,2),1);
solved_flag = zeros(size(X_matrix,2),1);
if save_flows
    X_bin_paths_cell = cell(size(X_matrix,2),1);
    t_paths_cell = cell(size(X_matrix,2),1);
    f_paths_cell = cell(size(X_matrix,2),1);
    arc_mask_paths_cell = cell(size(X_matrix,2),1);
end
flag_self_loops = cell(size(X_matrix,2),1);
% For each OD pair
for od_pair_idx = 1:size(X_matrix,2) 
% for od_pair_idx = 1:size(X_matrix,2) 
    X_copy_od = X_matrix(:,od_pair_idx);
    % --- Retrieve origin and destination nodes ---
    orig_node = find(D(:,od_pair_idx) < 0); % todo -- (find(D(:,od_pair_idx) == -1))
    dest_note = find(D(:,od_pair_idx) > 0); % todo -- (find(D(:,od_pair_idx) == 1))
    %%%%% Leo Fab changed
    epsl_od = -D(orig_node, od_pair_idx)/100;
    X_copy_od(X_copy_od<epsl_od) = 0;
%     if max(X_copy_od(X_copy_od>0))-min(X_copy_od(X_copy_od>0))>1e-3
%         1
%     end
    %X_copy_od(X_copy_od>0)
    %%%%%%
    % --- Assemble reduced network (with the arcs used in the solution X_matrix(:,od_pair_idx)) ---
%     arc_mask = X_matrix(:,od_pair_idx)>0; % logical mask of arcs used in solution X_matrix(:,od_pair_idx)
    arc_mask = X_copy_od>0; % logical mask of arcs used in solution X_matrix(:,od_pair_idx)
    arc_mask_idx = find(arc_mask == 1);
    inc_G_reduced = B(:,arc_mask); % reduced incidence matrix
    % --- Retrieve all paths from orig_node to dest_node with the arcs used in the solution X_matrix(:,od_pair_idx) ---
    % WARNING: edged ordering of G_reduced can be completely diffferent than the original order
    G_reduced = incidence2graph(inc_G_reduced);
%     flag = 1;
%     while flag
    % Find all self loops in the reduced graph (path solution from optimization)
    [~,edgecycles] = allcycles(G_reduced);%,'MaxNumCycles',10);
    if ~isempty(edgecycles)
        flag_self_loops{od_pair_idx,1} = true;
        for sl_idx = 1:length(edgecycles)
            self_loops = edgecycles{sl_idx,1};
            % Find the minimum flow in all the self loops (to make sure we don't remove edges that are part of the actually used  path)
            min_f_self_loops = min(X_copy_od(arc_mask_idx(self_loops)));
            X_copy_od(arc_mask_idx(self_loops)) = X_copy_od(arc_mask_idx(self_loops)) - min_f_self_loops;
        end
%         fprintf("Problem\n");
    end
%     if isempty(edgecycles)
%         flag = 0; 
%     else
%         arc_mask = X_copy_od>0; % logical mask of arcs used in solution X_matrix(:,od_pair_idx)
%         arc_mask_idx = find(arc_mask == 1);
%         inc_G_reduced = B(:,arc_mask); % reduced incidence matrix
%         % --- Retrieve all paths from orig_node to dest_node with the arcs used in the solution X_matrix(:,od_pair_idx) ---
%         % WARNING: edged ordering of G_reduced can be completely diffferent than the original order
%         G_reduced = incidence2graph(inc_G_reduced);
%     end
%     end
%     arc_mask = X_copy_od>1e-6; 
%     arc_mask_idx = find(arc_mask == 1);
%     inc_G_reduced = B(:,arc_mask); 
%     G_reduced = incidence2graph(inc_G_reduced);
    paths = allpaths(G_reduced,orig_node,dest_note);

    % --- Create reduced path vectors (bar{x} on the board) ---
    X_paths_G_reduced = zeros(length(arc_mask_idx),length(paths));
    for p_idx = 1:length(paths)
        for arc_idx = 1:length(arc_mask_idx)
            % 1 if path p_idx includes arc_idx
            % WARNING: arc indexing corresponds to the arc indexing in arc_mask
            % source and target nodes for each arc of the reduced G
            orig_arc_idx = find(B(:,arc_mask_idx(arc_idx)) == -1); % orig_arc_idx = find(inc_G_reduced(:,arc_idx) == -1); 
            dest_arc_idx = find(B(:,arc_mask_idx(arc_idx)) == 1); % dest_arc_idx = find(inc_G_reduced(:,arc_idx) == 1);
            if ~isempty(strfind(paths{p_idx,1},[orig_arc_idx dest_arc_idx])) % paths{p_idx}
                X_paths_G_reduced(arc_idx,p_idx) = 1;
            end
        end
    end
    
    % --- Compute path travel times ---
    t_paths = X_paths_G_reduced'*G.Edges.Weight(arc_mask);
    % t_paths = zeros(length(paths),1);
    % for p_idx = 1:length(paths)
    %    t_paths(p_idx) =  X_paths_G_reduced(:,p_idx)'*t_paths(arc_mask);
    % end
    % --- Assemble optimization problem (minimum accessibility) ---
    f =  sdpvar(length(paths),1); % Flows for each path
    % Constraints
    cnstr = [X_paths_G_reduced*f >= X_copy_od(arc_mask)-epsl_od/2,...
            X_paths_G_reduced*f <= X_copy_od(arc_mask)+epsl_od/2, f>=0];
    dt_max0 = t_paths-T_max;
    dt_max0(dt_max0<0) = 0;
    obj = dt_max0'*f;
    opts = sdpsettings('solver','gurobi','verbose',1,'showprogress', 1);
    sol = optimize(cnstr,obj,opts);
    opts.gurobi.FeasibilityTol = 1e-9;
    opts.gurobi.OptimalityTol = 1e-9;
    opts.gurobi.TuneTimeLimit = 0;
    if sol.problem 
        solved_flag = 1;
    end
    assert(~sol.problem);
    if save_flows
        X_bin_paths_cell{od_pair_idx,1} = X_paths_G_reduced;
        arc_mask_paths_cell{od_pair_idx,1} = arc_mask;
        t_paths_cell{od_pair_idx,1} = t_paths;
        f_paths_cell{od_pair_idx,1} = value(f);
    end

    minAcc_f = value(f);
%     minAcc(od_pair_idx) = dt_max0'*minAcc_f;
    minAcc(od_pair_idx) = (dt_max0/T_max).^2'*minAcc_f;

    % --- Assemble optimization problem (maximum accessibility) ---
%     f =  sdpvar(length(paths),1); % Flows for each path
%     % Constraints
%     cnstr = [X_paths_G_reduced*f == X_matrix(arc_mask,od_pair_idx), f>=0];
%     dt_max0 = t_paths-T_max;
%     dt_max0(dt_max0<0) = 0;
%     obj = -dt_max0'*f;
%     opts = sdpsettings('solver','gurobi','verbose',0);
%     optimize(cnstr,obj,opts);
%     maxAcc_f = value(f);
%     maxAcc(od_pair_idx) = dt_max0'*maxAcc_f;
% 
%     % --- Assemble optimization problem (minimum path flow variance) ---
%     f =  sdpvar(length(paths),1); % Flows for each path
%     % Constraints
%     cnstr = [X_paths_G_reduced*f == X_matrix(arc_mask,od_pair_idx), f>=0];
%     dt_max0 = t_paths-T_max;
%     dt_max0(dt_max0<0) = 0;
%     %mu = t_paths'*f;
%     %obj = (t_paths*sum(f)-mu)'*diag(f)*(t_paths*sum(f)-mu);
%     obj = norm(f,2);
%     opts = sdpsettings('solver','gurobi','verbose',0);
%     optimize(cnstr,obj,opts);
%     minVar_f = value(f);
%     minVarAcc(od_pair_idx) = dt_max0'*minVar_f;

    
%%%%%%%%%%%%%%%%
% % --- Print progress ---
%     fprintf("-----  OD-pair: (%d,%d)  -----\n",orig_node,dest_note);
%     fprintf("Paths:\n");
%     for p_idx = 1:length(paths)
%         fprintf("\t"); fprintf("%d ",paths{p_idx,1}'); fprintf("\n");
%     end
%     fprintf("Min. Acc.: %g (",minAcc(od_pair_idx)); fprintf(" %g ",minAcc_f');fprintf(")\n");
%%%%%%%%%%%%%%
    
    
    %     fprintf("Max. Acc.: %g (",maxAcc(od_pair_idx)); fprintf(" %g ",maxAcc_f');fprintf(")\n");
%     fprintf("Min. Var.: %g (",minVarAcc(od_pair_idx)); fprintf(" %g ",minVar_f');fprintf(")\n");
end

if max(solved_flag) == 0
    AFI = 60*sum(minAcc)/(sum(abs(D),'all')/2); % min
    t = G.Edges.Weight; 
    TT = 60*t'*X_matrix*ones(size(D,2),1)/(sum(sum(abs(D)))/2); % min
else
    TT = nan;
    AFI = nan;
end

% Save path flows info
if save_flows
% save(fp_save,'X_bin_paths_cell','arc_mask_paths_cell','t_paths_cell','f_paths_cell');
save(fp_save,'X_bin_paths_cell','arc_mask_paths_cell','t_paths_cell','f_paths_cell','flag_self_loops');
end

% if acc_flag
%     save('output/nodepc1/AFI_maxAcc_nodepc1_ncar4000_Tmax30.mat','AFI','TT');
% else
%     save('output/nodepc1/AFI_minTravTime_nodepc1_ncar4000.mat','AFI','TT');
% end
% 
% 
% return
% 
% TT =load('/home/mech001/20225929/Sara MSc/march22_run/output/nodepc1/AFI_minTravTime_nodepc1_ncar4000.mat');
% Acc  =load('/home/mech001/20225929/Sara MSc/march22_run/output/nodepc1/AFI_maxAcc_nodepc1_ncar4000_Tmax30.mat');
% [TT.AFI Acc.AFI]
% [TT.TT Acc.TT]

end
%% Auxiliary functions

function G = incidence2graph(I)
    [nodes_orig,~] = find(I == -1);
    [nodes_dest,~] = find(I == 1);
    G = digraph(nodes_orig,nodes_dest,ones(length(nodes_orig),1),size(I,1));
end


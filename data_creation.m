close all; clear; clc;
load ("variables\matlab_data_Eindhoven.mat")
load ("variables\matlab_data_od_paths.mat")

%%
adj                 = adj / 3600; % seconds to hours
times               = times / 3600; 
Demand              = (full_demand / 24)*2; % trips per hour aprox 57K

% weight in hours to traverse arcs
G                   = digraph(adj);
G.Nodes.Type        = nodes_type;

%%
% 1-car edge, 2-bike edge, 3-walking edge, 4-pt edge, 5-walk-pt edge
for i = 1:size(G.Edges,1)
    nodes = G.Edges.EndNodes(i,:);
    if G.Nodes.Type(nodes(1))=='c' & G.Nodes.Type(nodes(2))=='c'
        G.Edges.Type(i) = 1;
    elseif G.Nodes.Type(nodes(1))=='b' & G.Nodes.Type(nodes(2))=='b'
        G.Edges.Type(i) = 2;
    elseif G.Nodes.Type(nodes(1))=='w' & G.Nodes.Type(nodes(2))=='w'
        G.Edges.Type(i) = 3;
    elseif G.Nodes.Type(nodes(1))=='b' & G.Nodes.Type(nodes(2))=='w'
        G.Edges.Type(i) = 3;
    elseif G.Nodes.Type(nodes(1))=='w' & G.Nodes.Type(nodes(2))=='b'
        G.Edges.Type(i) = 3;
    elseif G.Nodes.Type(nodes(1))=='c' & G.Nodes.Type(nodes(2))=='w'
        G.Edges.Type(i) = 3;
    elseif G.Nodes.Type(nodes(1))=='w' & G.Nodes.Type(nodes(2))=='c'
        G.Edges.Type(i) = 3;
    elseif G.Nodes.Type(nodes(1))=='p' & G.Nodes.Type(nodes(2))=='w'
        G.Edges.Type(i) = 3;
    elseif G.Nodes.Type(nodes(1))=='o' & G.Nodes.Type(nodes(2))=='b'
        G.Edges.Type(i) = 3;
    elseif G.Nodes.Type(nodes(1))=='o' & G.Nodes.Type(nodes(2))=='c'
        G.Edges.Type(i) = 3;
    elseif G.Nodes.Type(nodes(1))=='o' & G.Nodes.Type(nodes(2))=='w'
        G.Edges.Type(i) = 3;
    elseif G.Nodes.Type(nodes(1))=='p' & G.Nodes.Type(nodes(2))=='c'
        G.Edges.Type(i) = 3;
    elseif G.Nodes.Type(nodes(1))=='w' & G.Nodes.Type(nodes(2))=='d'
        G.Edges.Type(i) = 3;
    elseif G.Nodes.Type(nodes(1))=='p' & G.Nodes.Type(nodes(2))=='p'
        G.Edges.Type(i) = 4;
    elseif G.Nodes.Type(nodes(1))=='w' & G.Nodes.Type(nodes(2))=='p'
        G.Edges.Type(i) = 5;
    elseif G.Nodes.Type(nodes(1))=='c' & G.Nodes.Type(nodes(2))=='p'
        G.Edges.Type(i) = 5;
    end
end

%%
G.Nodes.X = positions(:,1);
G.Nodes.Y = positions(:,2);

origins         = origin_nodes_ind;
destinations    = destination_nodes_ind;

B               = incidence(G); % Incidence matrix
nO              = length(origins); 
nOD             = length(destinations);
nArcs           = size(B,2); 
nNodes          = size(B,1);

%% % Assign pc code data to origins

pc_ = zeros(nNodes, 1);
for i = 1:nNodes
    if G.Nodes.Type(i)=='o'
        ind = find(origin_nodes_ind == i-1);
        pc = origin_pc(ind,:);
        if pc == '5633'
            pc = '5632';
        end
        pc_(i) = str2double(pc);
%     elseif G.Nodes.Type(i)=='d'
%         ind = find(destination_nodes_ind == i-1);
%         pc_(i) = deatination_pc(ind);
    end
end

G.Nodes.PC = pc_;
%% %Build D matrix of od-pairs and demand for flow optimization
% D = [];
% for i = 1:nO
%     for j = 1:nOD
%         % check which origins go to which destinations 
%         indOrigin = origins(i) + 1;                     
%         indDestination = destinations(j) + 1;
%         if i~=j & Demand(indOrigin,indDestination) > 0
%             d                   = zeros(nNodes,1); % start and end of trip
%             d(indOrigin)        = - Demand(indOrigin,indDestination); 
%             d(indDestination)   = Demand(indOrigin,indDestination); 
%             D                   = [D, d];
%         end
%     end
% end
% D1 = D;
D = [];
for i = 1:length(od_pairs)
    indOrigin           = od_pairs(1, i) + 1;
    indDestination      = od_pairs(2, i) + 1;
    d                   = zeros(nNodes,1); 
    d(indOrigin)        = - Demand(indOrigin, indDestination); 
    d(indDestination)   = Demand(indOrigin, indDestination); 
    D                   = [D, d];
end
% D2 = D;
%%

save("model\data_g.mat", ...
     "G", ...
     "D", ...
     "full_demand", ...
     "B", ...
     "nArcs", ...
     "nNodes")

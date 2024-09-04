clear; close all; clc;
load("model\data_g.mat");
load("model\data_shortPaths.mat");
% X_matrix = Xfast;
% X_matrix = Xslow;

% load('output/avgAcc.mat')
% X_matrix = sol_avgAcc.X;
z = zeros(nNodes,1);
type = G.Nodes.Type;

for node = 1:nNodes
    if type(node) == 'o'
        z(node) = 25;
    elseif type(node) == 'b'
        z(node) = 20;
    elseif type(node) == 'w'
        z(node) = 15;
    elseif type(node) == 'c'
        z(node) = 10;
    elseif type(node,:) == 'pt'
        z(node) = 5;
    elseif type(node) == 'd'
        z(node) = 0;
    end
end

adj_g = adjacency(G);

% ods = [612, 617, 637, 638, 698, 766, 795, 810, 910, 935, 958, 997];


% for i = 829%1:size(X_matrix,2) 
    f1 = plot(graph(adj_g+adj_g'),'XData',G.Nodes.X,'YData',G.Nodes.Y, 'ZData',z);
%     aux= G.Edges.EndNodes(X_matrix(:,i)>0,:)';
    aux= G.Edges.EndNodes(~G.Edges.Type,:)';
%     aux= G.Edges.EndNodes(arc_mask_idx(self_loops),:)';
    
    highlight(f1,aux(:),'NodeColor','g','EdgeColor','g','LineWidth',5)
%     highlight(f1,find(D(:,i)~=0),'NodeColor','r','MarkerSize',10)
%     fig_save = sprintf("figures/paths/paths_%d.fig",i);
%     savefig(fig_save)
%     i
%     pause()
% end



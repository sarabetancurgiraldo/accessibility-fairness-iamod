close all; clear; clc;

load("model\data_g.mat");
load("variables\matlab_data_od_paths.mat");
load("variables\matlab_data_Eindhoven.mat");

%% Flow matrices for shortest paths

nOD = length(od_pairs);

% Cell array containing shortest path between od-pair 
% shortestPaths{i,:} list of nodes in the path for od-pair i
% shortestPaths{:,1} full graph         (fast) 
% shortestPaths{:,2} graph w/out cars   (slow) 
shortestPaths = cell(nOD,2);

for i = 1:2 % 1-full graph, 2-noCar graph
    % Flow matrix | X(i,j) = 1 if path for od-pair j uses arc i 
    X = zeros(nArcs,nOD);
    if i == 1; paths = paths_dict.full; 
    elseif i == 2; paths = paths_dict.no_car; 
    end
    for od = 1:nOD
        len_path = size(paths{od},1);
        path = zeros(1,len_path);
        for j = 1:len_path
            node_name = strtrim(paths{od}(j,:));
            node_ind = find(ismember(nodes, node_name));
            path(j) = node_ind;
        end
        shortestPaths{od,i} = path;
        for j = 1:len_path-1
            indEdge = find(ismember(G.Edges.EndNodes, ...
                                    [path(j),path(j+1)],'rows'));
            X(indEdge,od) = 1;
        end
    end
    % Assign flow matrix | full graph --> Xfast | noCar graph --> Xslow
    if i == 1; Xfast = X; 
    elseif i == 2; Xslow = X; 
    end
end

%% Regions

pc = G.Nodes.PC;
pc_unique = unique(pc);
% Default value for not origin nodes = 0 (remove) 
pc_unique = pc_unique(~ismember(pc_unique,0)); 

% Build multiplier of population and selector matrix
nR = length(pc_unique); 

R_selector = zeros(nR, nOD);
for od = 1:nOD
    origin = find(D(:,od)<0);
    pc_origin = G.Nodes.PC(origin);
    R_i = find(pc_unique==pc_origin);
    R_selector(R_i, od) = 1;
end

% From data creation: Demand = (full_demand / 24)*2; 
%   D(o,i) = -Demand(o,d); D(d,i) = Demand(o,d) 
%       o, d = origin, destination for od-pair i

% 2.75 # of trips per day 
multiplier_population = 24/(2.75*2); 

% population_region - # of people
% divide by 2 to avoid double counting 
population_region = R_selector * multiplier_population*(sum(abs(D),1)'/2); 


save("model\data_shortPaths.mat", ...
     "shortestPaths", "od_pairs", "Xfast", "Xslow", "nOD", ...
     "pc_unique", "R_selector", "population_region")

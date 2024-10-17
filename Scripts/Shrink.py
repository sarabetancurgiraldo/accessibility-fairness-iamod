"""
@author: L.J. Koenders

Implementation of the Shrink compression algorithm.
"""
import numpy as np
import random as rnd

class Shrink:
    """Import graph to perform Shrink compression on."""
    def __init__(self, graph):
        """ Initialize variables."""
        self.graph = graph
        
        self.threshold = 1
        self.start_idx = 0
        self.Nu = self.Nuv = self.Nv = set()
        self.N = set()
        self.K = 0
        self.parent = {k:k for k in list(graph.nodes)}
        
    def Execute(self, iterations, version=1.3):
        """Execute Shrink algorithm.
        
        Parameters
        ----------
        iterations : int, float
            Choose how many mergings should take place.
            
            If given a float value, it is treated as compression ratio.
        
        version : float (default = 1.3)
            Choose node selection method.
            
        Output
        ------
        graph of input type."""
        
        if iterations%1!=0: # floats with decimal value of 0, such as 1.000, are treated as iterations
            iterations = int( len(self.graph) * iterations )
        
        for idx in range(iterations):
            U, V = self._NodeSelection(version)
            self._Merge(U, V)
        
        return self.graph
    
    def _Merge(self, U, V):
        # Check if U and V are neighbours
        if V not in self.graph.neighbors(U):
            raise Exception(f"Something went wrong: {U} and {V} are not neighbours.")

        self._NeighbourSets(U, V, [1,2]) # calculate neighbours
        for child in self.parent: # update parent info for "disappeared" node
            if self.parent[child] == V:
                self.parent[child] = U

        # Calculate cost per edge
        costs = self._CostsCi(U, V)
        
        # Perform normal Shrink procedure
        new_weights = self._CalculateWeights(costs, U, V)
        self._ImplementWeights(U, V, new_weights)
        
    def _NeighbourSets(self, U, V, sets):
        """
        Calculate neighbour sets for nodes U and V.
        
        Parameters
        ----------
        U, V : int, str
            Names of nodes U and V. Probably an integer, but you can name nodes anything.
        
        sets : int, list
            Indicate which neighbour sets you want to recalculate. 
            
            sets = 1 : N(u), N(uv), N(v)
            
            sets = 2 : N (set of all neighbours), K (len(N))
            
        Returns
        -------
        Returns Nu, Nuv, Nv, N and K to class.
        
        If U == V, N(u), N(uv), N(v) are emptied and N and K are recalculated.
        """
        if U != V:
            if sets == 1 or (type(sets) != int and 1 in sets):
                # Determine neighbour sets
                nbU = set(self.graph.neighbors(U)) # all neighbours of U
                nbV = set(self.graph.neighbors(V)) # all neighbours of V
        
                self.Nuv = nbU.intersection(nbV)
                self.Nu  = nbU - {V} - self.Nuv
                self.Nv  = nbV - {U} - self.Nuv
            if sets == 2 or (type(sets) != int and 2 in sets):
                self.N = self.Nu | self.Nv | self.Nuv
                self.K = len(self.N)
        else:
            self.Nu = self.Nuv = self.Nv = set()
            self.N = set(self.graph.neighbors(U))
            self.K = len(self.N)
    
    def _NodeSelection(self, version, exclude=None):
        """Execute node selection procedure.
        
        version : float
            Choose version of node selection. 
            
            1.x -> node selection method from Sadri paper
            
            2   -> random selection from all pairs that are not in exclude
            
        exclude : tuple, list
            Pair(s) of nodes to exclude from random selection in version 2."""
        if int(version) == 1:
            nodeList = list(self.graph._node.keys())#self._GetNodeList(self.graph)
            while True:
                # choose node 1
                try:
                    U = nodeList[self.start_idx]
                except:
                    self.start_idx = 0
                    continue
                
                # select node 2
                neighbours = set(self.graph.neighbors(U))
                min_weight = (0, np.Inf)
                for neighbour in neighbours:
                    weight = self.graph[U][neighbour]["weight"]
                    if weight < min_weight[1]:
                        min_weight = (neighbour, weight)
                V = int(min_weight[0])
                
                # Determine neighbour sets
                self._NeighbourSets(U, V, 1)
                
                # Evaluate candidate pair (U, V)
                # Version 1.3 is presumable used in the Shrink paper
                if (version == 1.1 and  len(self.Nu) * len(self.Nv) <= self.threshold) or\
                   (version == 1.2 and (len(self.Nu) + len(self.Nuv)) * (len(self.Nv) + len(self.Nuv)) < self.threshold) or\
                   (version == 1.3 and  len(self.Nu) + len(self.Nv) <= self.threshold):
                    return U, V
                else:
                    self.threshold += 1
                    self.start_idx += 1
        elif version == 2:
            """Choose a random node pair."""
            edgeList = self._GetEdgeList(self.graph)
            if type(exclude)==tuple:
                try: edgeList.remove(exclude)
                except:
                    raise Exception(f"Could not remove {exclude} from edge list.")
            elif type(exclude)==list:
                for pair in exclude:
                    try: edgeList.remove(pair)
                    except Exception as e:
                        print(e)
                        raise Exception(f"Could not remove {pair} from edge list.")
                U, V = rnd.choice(edgeList)
                return U, V
            
        elif version == 3:
            # Add any new node selection procedure.
            raise Exception("Version 3 is not yet available.")
            return U, V
    
    def _CostsCi(self, U, V):
        """Calculate costs Ci for all neighbours."""
        costs = dict()
        for vi in self.N:
            cost = 0
            
            # Sum for vj in Nu
            for vj in self.Nu-{vi}:
                if vi in self.Nu: # from Nu to Nu
                    cost += self.graph[U][vi]["weight"]
                    cost += self.graph[U][vj]["weight"]
                elif vi in self.Nv: # from Nv to Nu
                    cost += self.graph[vi][V]["weight"]
                    cost += self.graph[V][U]["weight"]
                    cost += self.graph[U][vj]["weight"]
                elif vi in self.Nuv: # from Nuv to Nu
                    cost += min(self.graph[U][vi]["weight"], 
                                self.graph[U][V]["weight"] + self.graph[V][vi]["weight"])
                    cost += self.graph[U][vj]["weight"]
                else:
                    raise Exception(f"Starting node ({vi}) not in neighbours set.")
            
            # Sum for vj in Nv
            for vj in self.Nv-{vi}:
                if vi in self.Nu: # from Nu to Nv
                    cost += self.graph[U][vi]["weight"]
                    cost += self.graph[U][V]["weight"]
                    cost += self.graph[V][vj]["weight"]
                elif vi in self.Nv: # from Nv to Nv
                    cost += self.graph[vi][V]["weight"]
                    cost += self.graph[V][vj]["weight"]
                elif vi in self.Nuv: # from Nuv to Nv
                    cost += min(self.graph[V][vi]["weight"], 
                                self.graph[U][V]["weight"] + self.graph[U][vi]["weight"])
                    cost += self.graph[V][vj]["weight"]
                else:
                    raise Exception(f"Starting node ({vi}) not in neighbours set.")
            
            # Sum for vj in Nuv
            for vj in self.Nuv-{vi}:
                if vi in self.Nu: # from Nu to Nuv
                    cost += self.graph[U][vi]["weight"]
                    cost += min(self.graph[U][vj]["weight"], 
                                self.graph[U][V]["weight"] + self.graph[V][vj]["weight"])
                elif vi in self.Nv: # from Nv to Nuv
                    cost += self.graph[V][vi]["weight"]
                    cost += min(self.graph[V][vj]["weight"], 
                                self.graph[U][V]["weight"] + self.graph[U][vj]["weight"])
                elif vi in self.Nuv: # from Nuv to Nuv
                    cost += min(self.graph[V][vj]["weight"] + self.graph[V][vi]["weight"],
                                self.graph[U][vj]["weight"] + self.graph[U][vi]["weight"],
                                self.graph[V][vj]["weight"] + self.graph[U][V]["weight"] + self.graph[U][vi]["weight"],
                                self.graph[V][vi]["weight"] + self.graph[U][V]["weight"] + self.graph[U][vj]["weight"])
                else:
                    raise Exception(f"Starting node ({vi}) not in neighbours set.")
            # Assign cost to vi
            costs[vi] = cost
        return costs

    def _CalculateWeights(self, costs, U, V):
        # Get a proper initial guess by using the original Shrink method
        new_weights = {}
        if self.K == 1:
            if len(self.Nu) == 1:
                new_weight = self.graph[U][list(self.Nu)[0]]["weight"] + self.graph[U][V]["weight"]/2
            elif len(self.Nv) == 1:
                new_weight = self.graph[V][list(self.Nv)[0]]["weight"] + self.graph[U][V]["weight"]/2
            elif len(self.Nuv) == 1:
                # new_weight = (self.graph[U][list(self.Nu)]["weight"] + self.graph[V][list(self.Nv)]["weight"])/2
                new_weight = (self.graph[U][list(self.Nuv)[0]]["weight"] + self.graph[V][list(self.Nuv)[0]]["weight"])/2
            else:
                raise Exception("Only one neighbour, but not in any neighbour set.")
            new_weights[[x for x in self.N][0]] = new_weight # weird formulation, but this extracts the node name from self.N
        else:
            for node in costs:
                if self.K > 2:
                    S = sum(costs.values()) / (2*self.K-2)
                    new_weight = (costs[node]-S)/(self.K-2)
                elif self.K == 2:
                    new_weight = costs[node]/2
                elif self.K != 1:
                    raise Exception("K=0, there are no neighbours.")
                new_weights[node] = new_weight
        return new_weights
    
    def _ImplementWeights_(self, U, V, new_weights, node_centering=True):
        coordList = self._GetCoordList(self.graph)
        if V != None:
            if node_centering == True:
                # Put new node in middle of U and V (for Mauro)
                xCoord = (coordList[U][0]+coordList[V][0])/2
                yCoord = (coordList[U][1]+coordList[V][1])/2
            else:
                xCoord = coordList[U][0]
                yCoord = coordList[U][1]
            self.graph.remove_nodes_from([U, V])
        else:
            # Just updating weights around U
            xCoord = coordList[U][0]
            yCoord = coordList[U][1]
            self.graph.remove_nodes_from([U])
        
        self.graph.add_node(U, x=xCoord, y=yCoord)
        for node in self.N:
            self.graph.add_weighted_edges_from([(U,node,new_weights[node])])
    
    def _GetEdgeList(self, graph):
        """
        Collect the edges in a graph, type list().
        """
        edgeList = []
        for B in graph._adj.keys():
            for E in graph._adj[B].keys():
                edgeList +=[(B,E)]
        return edgeList
    
    def _GetNodeList(self, graph):
        """
        Collect the nodes in a graph, type list().
        """
        return list(graph._node.keys())
    
    def _GetCoordList(self, graph):
        """
        Collect the coordinates of all nodes in a graph, type dict().
        Supported types: FlexibleGraph, DiGraph.
        """
        # graph = __ConvertFGToDiGraph(graph) # Take care of possible FlexibleGraph object type
        
        coordList = {}
        for idx in graph._node.keys():
            try:
                coordList[idx] = [graph._node[idx]['x'], graph._node[idx]['y']]
            except:
                idx
        return coordList
    def _ImplementWeights(self, U, V, new_weights, node_centering=True):
        flag = 0
        try: coordU = [self.graph._node[U]['x'], self.graph._node[U]['y']] 
        except: flag += 1
        try: coordV = [self.graph._node[V]['x'], self.graph._node[V]['y']] 
        except: flag += 2
        
        # coordList = self._GetCoordList(self.graph)
        if V != None:
            if node_centering == True:
                if flag == 0:
                    # Put new node in middle of U and V (for Mauro)
                    xCoord = (coordU[0]+coordV[0])/2
                    yCoord = (coordU[1]+coordV[1])/2
                elif flag == 1:
                    # No coordinates of node U
                    xCoord = coordV[0]
                    yCoord = coordV[1]
                elif flag == 2:
                    # No coordinates of node V
                    xCoord = coordU[0]
                    yCoord = coordU[1]
                elif flag == 3:
                    # No coordinates of node U and V
                    raise Exception(f"Node {U} and {V} both don't have coordinate information.")
        #     else:
        #         xCoord = coordList[U][0]
        #         yCoord = coordList[U][1]
            self.graph.remove_nodes_from([U, V])
        # else:
        #     # Just updating weights around U
        #     xCoord = coordList[U][0]
        #     yCoord = coordList[U][1]
        #     self.graph.remove_nodes_from([U])
        
        self.graph.add_node(U, x=xCoord, y=yCoord)
        for node in self.N:
            self.graph.add_weighted_edges_from([(U,node,new_weights[node])])
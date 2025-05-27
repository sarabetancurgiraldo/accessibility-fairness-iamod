# Run Instructions

## Network creation
- Install packages and modules from requirements.txt
- Change graph.py script from peartree module: Find "iteritems" and replace with "items". iteritems function from pandas changed.
- Run: Scripts/networksCreation.py
  * Parameters for run:
    + share_nodes: Take portion of nodes of region for OD-pairs (i.e. 0.25=25% of the nodes in the region).
    + max_nodes_pc: Minimum number of nodes to take per region for OD-pairs. 
    + average_speed_walk: Walking speed in m/s (5 km/h). 
    + average_speed_car: Walking speed in m/s (25 km/h). 
    + parking_time: Approximate waiting time to park. Time to traverse switching arc entering parking layer (36s). 
    + target_nodes: Expected number of nodes to result from shrinking networks.
    + factor_wb: For bike layer as copy of walking layer, bike speed is taken as a multiple of walking speed (3x).
    + north, east, south, west: Coordinates of studied area.
    + default_wait_pt: Approximate waiting time to board public transportation ride. Time to traverse switching arc entering public transportation layer (4 minutes).
    + modes: Mode share from CBS data. Used for number of daily trips.

  * Variables for run:
    + flag_bike: Flag to have bike graph as walking graph copy (True default).
    + flag_parking: Flag to have a parking graph (False default).
    + flag_create_pt: Flag to create from scratch PT graph (False default -data saved as pickle file from previous executions-).
    + flag_load: Flag to load B, W, C from previous execution (True default -data saved as pickle file from previous executions-). 

- Run: Scripts/shortestPaths_toMatlab.py 
  To find shortest paths for each OD-pair with and without car.

## Data to Matlab
- Run: data_creation.m and data_shortPaths.m to get model to use in matlab optimizations

## Run optimizations
- Run: run_results.m runs optimizations and path allocation
- Possibility to run from HPC using jobs_run_results.sh 
  * Connect to ssh with username and password 
    + Under tue network connection run ssh username (i.e.: 20222295@hpc.tue.nl) and type password
    + Navigate to folder with scripts and run jobs: sbatch jobs_run_results.sh
    + Check queue: squeue

## Plot modal share

- Run: run_bar_plots.m 

## Results to Python

- Run: results_toPyhton.m 

## Plot unfairness level in python

- Run: Scripts/plot_ur.py


### Good luck!

# Thermal-aware energy-efficient task scheduling for DVFS-enabled data centers using `cvxpy`

This project is an optimal task scheduler that minimizes a data center's computation and AC energy consumption while
accounting for the heterogeneous thermal correlation among servers and using `python`+`cvxpy`

This is implementation is based on research paper [Thermal-aware energy-efficient task scheduling for DVFS-enabled data centers
Dong Han;Tao Shu 2015](https://ieeexplore-ieee-org.mutex.gmu.edu/document/7069401/) and the Python translation for [Data-center-energy-optimization](https://github.com/minicpp/Data-center-energy-optimization)

## Dependencies

```
numpy
ecos==2.0.5
cmake
scs==2.1.3
osqp==0.6.1
cvxpy
```

## Install
This installation should work on Apple M1 chip
```
python3 -m venv energy-optimizer-env
source energy-optimizer-env/bin/activate
pip3 install -r requirements.txt
```


## Project structure
- `dc_energy_input.json` all the machines and datacenter constants:
  - number_of_machines
  - ac_supply_temperature
  - red_temperature
  - air_density
  - flow_speed
  - heat_capacity
  - airflow_rec_correlation
  - number_of_tasks
  - task_req_mhz
  - machine_max_frequency
  - cpu_max_power
  - cpu_idle_power
  - cpu_heat_capacity
  - cpu_mass
- `energy_optimizer.py` the project implementation to optimize the energy consumption

## Run
```
python energy_optimizer.py
```

## Sample Results
```
Number of scheduled jobs on each machine:

[[1.]
 [1.]
 [1.]
 [1.]
 [1.]]

Tasks Computation Power:

[[85.65441608]
 [85.65441608]
 [85.65441608]
 [85.65441608]
 [85.65441608]]

AC supply temperature:

T_sup: [[16.79629581]]

Inlet temperature:

[[16.79629581]
 [16.79629581]
 [18.        ]
 [18.        ]
 [16.79629581]]

CPU temperature:

[[18.20212591]
 [18.20212591]
 [19.40583009]
 [19.40583009]
 [18.20212591]]

Outlet temperature:

[[18.        ]
 [18.        ]
 [19.20370418]
 [19.20370418]
 [18.        ]]

Outlet airflow power:

[[1280.86244979]
 [1280.86244979]
 [1366.51686587]
 [1366.51686587]
 [1280.86244979]]

Power of AC:

[[142.70891566]]

Total Computation Power:

[428.27208041]

Total power (AC+CMP):

[[570.98099607]]

```

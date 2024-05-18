import json
import numpy as np
from numpy.linalg import inv
import cvxpy as cp


def ind2sub(array_shape, ind):
    rows = (ind.astype("int32") // array_shape[1])
    cols = (ind.astype("int32") % array_shape[1])
    return rows, cols


def energy_optimizer():
    input_file = open("dc_energy_input.json", "r")
    dc_energy_input = json.loads(input_file.read())
    m = dc_energy_input["number_of_machines"]
    t_red = dc_energy_input["red_temperature"]  # The maximum temperature allowed in the datacenter
    c = dc_energy_input["heat_capacity"]   # J/g the number of heat units needed to raise the temperature of one gram of air by one degree.
    p = dc_energy_input["air_density"]  # g/m^3 the mass of one gram per one meter cube volume
    s = dc_energy_input["flow_speed"]   # m^3/s one meter cube volume of air flowing per second
    A = dc_energy_input["airflow_rec_correlation"]  # air recirculation correlation matrix based on the datacenter layout
    base_freq = 0

    # the specs based on the research paper simulated data using AMD FX-9590
    # The CPU voltage and frequency vary from 0.6 V to 1.5125 V and from 400 MHz to 4700MHz
    cpu_max_freq = dc_energy_input["machine_max_frequency"]
    cpu_max_power = dc_energy_input["cpu_max_power"]
    cpu_idle_power = dc_energy_input["cpu_idle_power"]
    cpu_mass = dc_energy_input["cpu_mass"]  # the mass of the CPU
    cpu_heat_capacity = dc_energy_input["cpu_heat_capacity"]  # the heat capacity of the CPU material
    task_req_mhz = dc_energy_input["task_req_mhz"]  # the mhz required to complete the task
    task_n = dc_energy_input["number_of_tasks"]  # the number of tasks to schedule

    mass_of_air_in = p * s  # g/s mass of flowing air in for each fan
    rec_coefficient = np.transpose(np.sum(A, axis=0, keepdims=True))  # total recirculated air for each machine
    AT = np.transpose(A)  # transforming summation of rec coefficients to a matrix
    rev_EAT = inv(np.eye(m) - AT)  # (E - A')^(-1)

    # (1 - rec_coefficient) subtracting the recirculated from the total air to get the AC air
    m_sup_array = np.multiply(mass_of_air_in, (1 - rec_coefficient))  # mass of air flowing from AC for each unit

    mass_of_air_out = np.multiply(mass_of_air_in, np.ones((m, 1)))  # mass of air flowing out from all the machines fans
    alpha = c * m_sup_array  # c * p * s for each machine fan

    c_0 = cpu_idle_power ** 0.5  # the square root of power consumption of an idle machine

    c_1 = (cpu_max_power ** 0.5 - c_0) ** 0.5 / cpu_max_freq  # the sqrt(cpu_power) needed for one mhz

    P_red = np.multiply(t_red * c * mass_of_air_in, np.ones((m, 1)))  # Q = Tcps the power consumption for T_red

    X_max = cpu_max_power ** 0.5  # the square root of power consumption of a machine at max frequency

    job_allocated = np.zeros((m, 1))

    while task_n >= 0:
        task = task_req_mhz * np.ones((task_n, 1))
        freq_allocated = task_req_mhz * job_allocated

        x = cp.Variable([m, 1])  # total power consumption for each task based on the adjusted frequency
        P_e = cp.Variable([m, 1])  # machines CPU consumed power
        COP = cp.Variable([1, 1])  # Coefficient of Performance for the AC
        P_sup = cp.Variable([m, 1])  # the supplied power based on the supplied temperature
        T_sup = cp.Variable([1, 1])  # the supplied temperature from the AC
        f = cp.Variable([m, 1])  # the adjusted frequency for each machine

        # every time we optimize and schedule a task we reduce the number of tasks remaining
        if task_n > 0:
            y = cp.Variable([m, task_n])   # the scheduling matrix

        P_out = cp.Variable([m, 1])  # the total power of the air out

        objective = cp.Minimize(cp.quad_over_lin(x, COP) + cp.sum(P_e))  # minimizing the sum of AC power and CPU power

        constraints = [x >= cp.power(c_1 * f, 2) + c_0,
                       x <= X_max,
                       P_e >= cp.power(x, 2),
                       P_e <= cpu_max_power,
                       T_sup * alpha == P_sup,  # alpha is c*p*s we are calculating P = Tcps
                       T_sup >= 5.5,
                       COP == 0.265 * T_sup - 1.45,
                       P_sup >= 0,
                       P_out >= 0,
                       rev_EAT @ (P_sup + P_e) == P_out,
                       P_sup + AT @ P_out <= P_red]

        if task_n > 0:
            constraints.append(f == y @ task + freq_allocated + base_freq)  # the frequency needed for a task
            constraints.append(f <= cpu_max_freq)  # the frequency for any task should be less than cpu max frequency
            constraints.append(cp.sum(y) == 1)  # one task per machine
            constraints.append(y >= 0)  # either 0
            constraints.append(y <= 1)  # or 1 (scheduled)
        else:
            constraints.append(f == freq_allocated)

        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        # scheduling the job by rounding the max value to 1
        if task_n > 0:
            y2 = y.value
            v, i = y2.flatten().max(0), y2.flatten().argmax(0)  # max value and its index
            flag_solve = 0
            while v != - 100:
                i, j = ind2sub(y2.shape, i)   # based on the index and the matrix shape get the row, column values
                job_num = job_allocated[i] + 1  # schedule the job
                if job_num * task_req_mhz + base_freq > cpu_max_freq:  # above max frequency
                    y2[i, j] = - 100
                    v, i = y2.flatten().max(0), y2.flatten().argmax(0)
                else:
                    job_allocated[i] = job_num
                    flag_solve = 1
                    break

            if flag_solve == 0:
                print('\nError:Tasks exceed the capacity of Data Center\n')
                break
        task_n = task_n - 1

    print('\nNumber of scheduled jobs on each machine:\n')
    print(job_allocated)
    print('\nTasks Computation Power:\n')
    p_e = (c_0 + (c_1 * (task_req_mhz * job_allocated + base_freq)) ** 2) ** 2
    print(p_e)
    print('\nAC supply temperature:\n')
    p_out = np.matmul(rev_EAT, (P_sup.value + p_e))
    print("T_sup:", T_sup.value)
    print('\nInlet temperature:\n')
    T_in = (P_sup.value + np.matmul(AT, p_out)) / (c * (m_sup_array + np.matmul(AT, mass_of_air_out)))
    print(T_in)
    print('\nCPU temperature:\n')
    T_cpu = T_in + p_e / (cpu_heat_capacity * cpu_mass)
    print(T_cpu)
    print('\nOutlet temperature:\n')
    T_out = p_out / (c * mass_of_air_out)
    print(T_out)
    print('\nOutlet airflow power:\n')
    print(p_out)
    print('\nPower of AC:\n')
    P_AC = sum(p_e) / COP
    print(P_AC.value)
    print('\nTotal Computation Power:\n')
    P_CMP = sum(p_e)
    print(P_CMP)
    print('\nTotal power (AC+CMP):\n')
    P_TOTAL = P_AC + P_CMP
    print(P_TOTAL.value)

    print("\n Optimized Answer:", result)


if __name__ == '__main__':
    energy_optimizer()

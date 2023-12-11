# Continuous Time RL with LMUs
The recommended order for reading the notebooks is "RL with LMUs", "TD(0) with LMUs", "TD(theta) with LMUs", and then "TD(lambda) with LMUs".

Requirements:
 -  numpy, scipy, nengo, matplotlib
 
Code: 
 -  *lmu_networks*: Nengo networks for implementing LMUs (some using nengo.Processes, others with nengo.Ensembles). Also includes versions where the time window can be controlled with a input signal 
 -  *online_rl_networks*: Nengo networks that use LMUs to implement different TD updates (options are 'TD0', 'TDtheta', 'TDlambda'). Need the states and rewards over time as input. Currently there are no actor networks for action selection.
    -  ValueCritic: learns value function, using LMUProcess (i.e., perfect LMU dynamics)
    -  NeuralValueCritic: learns value function, using LMUNetwork (i.e., spiking neural LMU / approximate dynamics)
    -  SRCritic: learns successor features (state input must be given as the features) and a linear reward function, using LMUProcess (i.e., perfect LMU dynamics)
-  *offline_erl_networks.py*: Nengo networks that use LMUs to implement different TD updates for value function learning. Unlike the above networks, these update via replay through memories. Still a work in progress
      -  ValueCritic: learns value function, using LMUProcess (i.e., perfect LMU dynamics)
 -  *learning_rules*: Nengo learning rules DPES ('delayed PES') and SynapticModulation. DPES is used for the TD updates that use LMUs in *online_rl_networks.py*. SynapticModulation is used for the time window control of LMUs implemented via ensembles in *lmu_networks*.

Notebooks:
 -  *LMU with changing theta*: Examples of using each of the networks/processes from *lmu_networks.py*
 -  *RL with LMUs*: Explains some theory, motivation, and notation used in other notebooks
 -  *TD(0) with LMUs*: Examples of using the ValueCritic network from *online_rl_networks.py*, with the TD(0) learning rule
 -  *TD(theta) with LMUs*: Examples of using the ValueCritic network from *online_rl_networks.py*, with the TD(theta) learning rule. Also, one example that uses the NeuralValueCritic network
 -  *TD(theta) - learning with replay*: Examples of using the ValueCritic network from *offline_rl_networks.py*, with the TD(theta) learning rule.
 -  *TD(lambda) with LMUs*: Examples of using the ValueCritic network from *online_rl_networks.py*, with the TD(lambda) learning rule
 -  *Tabular example*: Simple example that uses the *get_critic_transforms* function from *online_rl_networks.py* to learn with LMUs without using nengo networks
 -  *Trace conditioning task*: Uses the ValueCritic network from *online_rl_networks.py*, with the TD(0) and TD(theta) learning rules, on a simple trace conditioning task
 -  *Tmaze*: : Uses the ValueCritic network from *online_rl_networks.py*, with the TD(theta) learning rule, on a simple T-maze task


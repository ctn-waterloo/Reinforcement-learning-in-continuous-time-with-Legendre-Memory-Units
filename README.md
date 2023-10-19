# Continuous Time RL with LMUs
The recommended order for reading the notebooks is "RL with LMUs", "TD(0) with LMUs", "TD(theta) with LMUs", and then "TD(lambda) with LMUs".
 
Code: 
 -  *ldn_networks.py*: Nengo networks for implementing LMUs. This includes a copy of Terry's code from learn_dyn_sys, where this is done via nengo.Processes. And there are versions where this is done with nengo.Ensembles. Also includes versions of both where the time window can be controlled with a input signal 
 -  *rl_networks.py*: Nengo networks that use LMUs to implement different TD updates (options are 'TD0', 'TDtheta', 'TDlambda'). Need the states and rewards over time as input. Outputs the values function over time. Currently there are no actor networks for action selection.
 -  *learning_rules.py*: Nengo learning rules DPES ('delayed PES') and SynapticModulation. DPES is used for the TD updates that use LMUs in *rl_networks.py*. SynapticModulation is used for the time window control of LMUs implemented via ensembles in *lmu_networks.py*.

Notebooks/ example useage:
 -  *LMU with changing theta.ipynb*: Examples of using each of the networks/processes from *lmu_networks.py*
 -  *RL with LMUs.ipynb*: Explains some theory, motivation, and notation used in other notebooks
 -  *TD(0) with LMUs.ipynb*: Examples of using the ValueCritic network, from *rl_networks.py*, with the TD(0) learning rule. Also, one example that uses the NeuralValueCritic network
 -  *TD(theta) with LMUs.ipynb*: Examples of using the ValueCritic network, from *rl_networks.py*, with the TD(theta) learning rule
 -  *TD(lambda) with LMUs.ipynb*: Examples of using the ValueCritic network, from *rl_networks.py*, with the TD(lambda) learning rule


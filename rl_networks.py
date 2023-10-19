import nengo
from nengo.network import Network
import numpy as np
import scipy.linalg
from scipy.special import legendre
import scipy.integrate as integrate

from learning_rules import DPES
from lmu_networks import LMUProcess, LMUNetwork_v2, LMUModulatedNetwork_v2

# Helpful util
def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))

##############################
# General network that learns a value function given LDNs (in nodes) & decoders
#######################

class ValueCritic(nengo.Network):
    def __init__(self,n_neurons_state, n_neurons_value, theta, d, discount, q_a, q_r, q_v,
                 algor,  learning_rate=1e-4, T_test=10000,state_encoders=None,tau=0.05, lambd=0.8,
                  **kwargs):
        super().__init__(**kwargs)
        
        self.activity_lmu_transform, self.reward_lmu_transform, self.value_transform, self.value_lmu_transform = get_RL_decoders(algor, discount, n_neurons_state, theta,
                                                     q_a=q_a, q_r = q_r, q_v=q_v, lambd=lambd)


        if algor=='TDLambda':
            pre_act_input_size=q_a
        else:
            pre_act_input_size=1
        with self:
            
            self.state_input = nengo.Node(size_in=d)
            if state_encoders is not None:
                self.state = nengo.Ensemble(n_neurons_state, d, encoders = state_encoders, intercepts=nengo.dists.Choice([sparsity_to_x_intercept(d, 0.1)]) )
            else:
                self.state = nengo.Ensemble(n_neurons_state, d, intercepts=nengo.dists.Choice([sparsity_to_x_intercept(d, 0.1)]))
            nengo.Connection(self.state_input, self.state)
            lmu_s = LMUProcess(theta=theta, q=q_a,size_in=n_neurons_state)
            self.state_memory = nengo.Node(lmu_s)
            nengo.Connection(self.state.neurons, self.state_memory,synapse=tau)

            self.reward_input = nengo.Node(size_in=1)
            lmu_r = LMUProcess(theta=theta, q=q_r,size_in=1)
            self.reward_memory = nengo.Node(lmu_r)
            nengo.Connection(self.reward_input, self.reward_memory, synapse=None) 

            lmu_v = LMUProcess(theta=theta, q=q_v,size_in=1)
            self.value = nengo.Ensemble(n_neurons_value,1)
            self.value_memory = nengo.Node(lmu_v)
            nengo.Connection(self.value, self.value_memory, synapse=tau)
            self.learn_connV = nengo.Connection(self.state.neurons, self.value, 
                                                transform=np.zeros((1,n_neurons_state)), 
                                        learning_rule_type = DPES(1,n_neurons_state, 
                                                                pre_act_input_size, 
                                                                learning_rate = learning_rate),synapse=tau)

            self.error = nengo.Node(lambda t,x: x if t<T_test else 0, 
                               size_in=(1 + n_neurons_state)*pre_act_input_size)
            
            nengo.Connection(self.reward_memory, self.error[:pre_act_input_size], 
                             transform=-self.reward_lmu_transform)
            nengo.Connection(self.value, self.error[:pre_act_input_size], 
                             transform=-self.value_transform, synapse=0.05)
            nengo.Connection(self.value_memory, self.error[:pre_act_input_size],  
                             transform=self.value_lmu_transform, synapse=None)
            nengo.Connection(self.state_memory, self.error[pre_act_input_size:], 
                             transform=self.activity_lmu_transform, synapse=None)
            
            nengo.Connection(self.error, self.learn_connV.learning_rule, synapse=None)
            self.rule = self.learn_connV.learning_rule

            
class NeuralValueCritic(Network):
    def __init__(self,n_neurons_state, n_neurons_value, n_neurons_lmus, theta, d, discount, q_a, q_r, q_v,
                 algor, learning_rate=1e-4, T_test=10000,state_encoders=None,tau=0.05,lambd=0.8,
                  **kwargs):
        super().__init__(**kwargs)
        
        self.activity_lmu_transform, self.reward_lmu_transform, self.value_transform, self.value_lmu_transform = get_RL_decoders(algor, discount, n_neurons_state, theta,
                                                     q_a=q_a, q_r = q_r, q_v=q_v, lambd=lambd)

        if algor=='TDlambda':
            pre_act_input_size=q_a
        else:
            pre_act_input_size=1
        with self:
            
            self.state_input = nengo.Node(size_in=d)
            if state_encoders is not None:
                self.state = nengo.Ensemble(n_neurons_state, d, encoders = state_encoders, intercepts=nengo.dists.Choice([sparsity_to_x_intercept(d, 0.1)]))
            else:
                self.state = nengo.Ensemble(n_neurons_state, d, intercepts=nengo.dists.Choice([sparsity_to_x_intercept(d, 0.1)]))
            nengo.Connection(self.state_input, self.state)
            self.state_memory = LMUNetwork_v2(n_neurons_lmus, theta=theta, q=q_a, size_in=n_neurons_state, tau=tau)
            nengo.Connection(self.state.neurons, self.state_memory.input, synapse=tau, transform=1/1000)

            self.reward_input = nengo.Node(size_in=1)
            self.reward_memory  = LMUNetwork_v2(n_neurons_lmus, theta=theta, q=q_r, size_in=1, tau=tau)
            nengo.Connection(self.reward_input, self.reward_memory.input, synapse=0) 

            self.value = nengo.Ensemble(n_neurons_value,1, radius=0.55)
            self.value_memory = LMUNetwork_v2(n_neurons_lmus, theta=theta, q=q_v, size_in=1, tau=tau)
            nengo.Connection(self.value, self.value_memory.input, synapse=0.05)
            self.learn_connV = nengo.Connection(self.state.neurons, self.value, transform=np.zeros((1,n_neurons_state)), 
                                           learning_rule_type=DPES(1,n_neurons_state,pre_act_input_size,learning_rate=1000*learning_rate), synapse=tau)

            self.error = nengo.Node(lambda t,x: x if t<T_test else 0, 
                               size_in=(1 + n_neurons_state)*pre_act_input_size)
            #print(pre_act_input_size)
            
            nengo.Connection(self.reward_memory.output, self.error[:pre_act_input_size], 
                             transform= -self.reward_lmu_transform)
            nengo.Connection(self.value, self.error[:pre_act_input_size], 
                             transform=-self.value_transform, synapse=0.05)
            nengo.Connection(self.value_memory.output, self.error[:pre_act_input_size],  
                             transform=self.value_lmu_transform, synapse=0.05)
            nengo.Connection(self.state_memory.output, self.error[pre_act_input_size:], 
                             transform=self.activity_lmu_transform, synapse=0.05)
            
            nengo.Connection(self.error, self.learn_connV.learning_rule, synapse=None)
            self.rule = self.learn_connV.learning_rule
            
            
   
            

#Function to return decoders need for LMU RL rules (TD0, TDtheta, TDlambda)
#( these are descibed below)

# The decoders returned are pre_activty_decoders, state_decoders, function_decoders, function_memory_decoders
# 
# The DPES rule will recieve input pre_activity and error (in a single vector)
# The update to state decoders will be of the form,
#    pre_activity @ error.T
# pre_activity is computed as
#    pre_activty_decoders @ A
# where A is either decoded activities or an LMU representation of activites depending on the rule
# The error is computed with three terms,
#    function_memory_decoders @ F - function_decoders @ f - state_decoders @ S 
# where f is the current value of the function we are learning
# F is the LMU representation of that function
# The function we want to learn is a discounted sum/integral of some variables that we'll refer to gernally as the state
#  S is the LMU representation of the state
# For value leanring the state is the reward
# For SR learning the state is the env state



def get_RL_decoders(rule_type, discount, n_neurons, theta, size=1,
                    q_a=10, q_r = 10, q_v=10, alpha=10, lambd=0.8):
    
    if rule_type=="TD0":
        activity_lmu_transform = np.asarray([legendre(i)(1) for i in range(q_a)])
        activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform.reshape(q_a, -1).T)

        reward_lmu_transform = np.asarray([legendre(i)(1) for i in range(q_r)])
        reward_lmu_transform = np.kron(np.eye(size), reward_lmu_transform.reshape(q_r, -1).T)
        
        value_transform = discount/theta
        
        value_lmu_transform = np.asarray([legendre(i)(1) for i in range(q_v)])
        value_lmu_transform = (1+discount/theta)*np.kron(np.eye(size), value_lmu_transform.reshape(q_v, -1).T)

        
    elif rule_type=="TDtheta":
        reward_lmu_transform = np.zeros(q_r)
        for i in range(q_r):
            intgrand = lambda x: (discount**(theta*(1-x)))*legendre(i)(2*x-1)
            reward_lmu_transform[i]= integrate.quad(intgrand, 0,1)[0]
    
        reward_lmu_transform =  np.kron(np.eye(size), reward_lmu_transform.reshape(1, -1))
        
        activity_lmu_transform = np.asarray([legendre(i)(1) for i in range(q_a)])
        activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform.reshape(q_a, -1).T)

        value_transform = discount**theta

        value_lmu_transform = np.asarray([legendre(i)(1) for i in range(q_v)])
        value_lmu_transform = np.kron(np.eye(size), value_lmu_transform.reshape(q_v, -1).T)

    elif rule_type=="TDlambda":
        reward_lmu_transform = np.zeros(q_r)
        for i in range(q_r):
            intgrand = lambda y,x: np.exp(-lambd*x*theta)*(discount**(theta*(1-y)))*legendre(i)(2*y-1)
            reward_lmu_transform[i]=integrate.dblquad(intgrand, 0,1,lambda x: x, lambda x: 1)[0]
        reward_lmu_transform =  np.kron(np.eye(size), reward_lmu_transform.reshape(1, -1))
        
        activity_lmu_transform = np.asarray([legendre(i)(1) for i in range(q_a)])
        activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform.reshape(q_a, -1).T)

        value_transform = 0

        value_lmu_transform1 = np.zeros(q_v)
        for i in range(q_v):
            intgrand = lambda x: np.exp(-lambd*x*theta)*(discount**(theta*x))*legendre(i)(2*x-1)
            value_lmu_transform1[i]=integrate.quad(intgrand, 0, 1)[0]
        
        value_lmu_transform2 = np.asarray([legendre(i)(1) for i in range(q_v)])
        value_lmu_transform = np.kron(np.eye(size), (value_lmu_transform1-value_lmu_transform2).reshape(q_v, -1).T)

        
    elif rule_type=="TDLambda":
        activity_lmu_transform = 1
        reward_lmu_transform = np.zeros((q_r, q_a))
        value_transform = np.zeros(q_a)
        value_lmu_transform = np.zeros((q_v, q_a))
        
        
        for i in range(q_a):
            intgrand = lambda x: (discount**(x*theta))*legendre(i)(2*x-1)
            value_transform[i]=integrate.quad(intgrand, 0,1)[0]

            for j in range(q_r):
                intgrand = lambda y,x: (discount**(theta*(x-y)))*legendre(i)(2*x-1)*legendre(i)(2*y-1)
                reward_lmu_transform[j,i]=integrate.dblquad(intgrand, 0,1,lambda x: 0, lambda x: x)[0]

        for i in range(np.min([q_v,q_a])):
            intgrand = lambda x: legendre(i)(2*x-1)**2
            value_lmu_transform[i,i]=integrate.quad(intgrand, 0, 1)[0]
            
        value_transform =  np.kron(np.eye(size), value_transform.reshape(-1,1))
        value_lmu_transform = np.kron(np.eye(size), value_lmu_transform.T)
        reward_lmu_transform = np.kron(np.eye(size), reward_lmu_transform.T)
    else:
        print("Not a valid rule")
    return activity_lmu_transform, reward_lmu_transform, value_transform, value_lmu_transform 
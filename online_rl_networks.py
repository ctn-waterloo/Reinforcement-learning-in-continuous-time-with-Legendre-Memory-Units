import nengo
from nengo.network import Network
import numpy as np
import scipy.linalg
from scipy.special import legendre, eval_legendre, eval_sh_legendre, lpn
import scipy.integrate as integrate

from learning_rules import DPES
from lmu_networks import LMUProcess, LMUNetwork_v2, LMUModulatedNetwork_v2

def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))



# General network that learns a value function given LMUs (in nodes)
class ValueCritic(nengo.Network):
    def __init__(self,n_neurons_state, n_neurons_value, theta, d, discount, q_n, q_r, q_v,
                 algor,  learning_rate=1e-4, T_test=10000,state_ensembles=None,tau=0.05, lambd=0.8,
                  **kwargs):
        super().__init__()
        
        self.activity_lmu_transform, self.reward_lmu_transform, self.value_transform, self.value_lmu_transform = get_critic_transforms(algor, discount, n_neurons_state, theta,
                                                     q_a=q_n, q_r = q_r, q_v=q_v, lambd=lambd)

        if algor=='TDLambda':
            pre_act_input_size=q_n
        else:
            pre_act_input_size=1
        with self:
            self.reset = nengo.Node(size_in=1)
            self.state_input = nengo.Node(size_in=d)
            if state_ensembles is not None:
                self.state = state_ensembles[0]
                self.state_memory = state_ensembles[1]
            else:
                self.state_input = nengo.Node(size_in=d)
                self.state = nengo.Ensemble(n_neurons_state, d, **kwargs)
                nengo.Connection(self.state_input, self.state, synapse=None)
                lmu_s = LMUProcess(theta=theta, q=q_n, size_in=n_neurons_state,with_resets=True)
                self.state_memory = nengo.Node(lmu_s)
                nengo.Connection(self.state.neurons, self.state_memory[1:],synapse=tau)
                nengo.Connection(self.reset, self.state_memory[0],synapse=None)

            self.reward_input = nengo.Node(size_in=1)
            lmu_r = LMUProcess(theta=theta, q=q_r,size_in=1,with_resets=True)
            self.reward_memory = nengo.Node(lmu_r)
            nengo.Connection(self.reward_input, self.reward_memory[1:], synapse=None) 
            nengo.Connection(self.reset, self.reward_memory[0],synapse=None)

            lmu_v = LMUProcess(theta=theta, q=q_v,size_in=1,with_resets=True)
            self.value = nengo.Ensemble(n_neurons_value,1)
            self.value_memory = nengo.Node(lmu_v)
            nengo.Connection(self.value, self.value_memory[1:], synapse=tau)
            nengo.Connection(self.reset, self.value_memory[0],synapse=None)
            
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
                 algor, learning_rate=1e-4, T_test=10000,state_ensemble=None,tau=0.05,lambd=0.8,
                  **kwargs):
        super().__init__()
        
        self.activity_lmu_transform, self.reward_lmu_transform, self.value_transform, self.value_lmu_transform = get_critic_transforms(algor, discount, n_neurons_state, theta,
                                                     q_a=q_a, q_r = q_r, q_v=q_v, lambd=lambd)

        if algor=='TDlambda':
            pre_act_input_size=q_a
        else:
            pre_act_input_size=1
        with self:
            
            self.state_input = nengo.Node(size_in=d)
            if state_ensemble is not None:
                self.state = state_ensemble
            else:
                self.state = nengo.Ensemble(n_neurons_state, d, **kwargs)
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
            
            

# General network that learns a SR function using LMUs in nodes 
class SRCritic(Network):
    def __init__(self,n_neurons_state, n_neurons_sr,n_neurons_r, theta, d, discount, q_n, q_s, q_sr, 
                 algor, T_test = 100000,state_ensembles=None, 
                 reward_learning_rate=5e-3, sr_learning_rate=1e-3,tau=0.05,
                  **kwargs):
        super().__init__()
        state_act_decoders, state_decoders, sr_decoders, sr_memory_decoders = get_critic_transforms(algor,
                                                            discount, n_neurons_state, theta, size=d, 
                                                            q_a=q_n, q_v = q_s, q_r=q_sr)
        with self:
            self.reset = nengo.Node(size_in=1)
            # State obs, an input
            self.state_input = nengo.Node(size_in=d)
            # Population representing the state
            if state_ensembles is not None:
                self.state = state_ensembles[0]
                self.state_memory = state_ensembles[1]
            else:
                self.state_input = nengo.Node(size_in=d)
                self.state = nengo.Ensemble(n_neurons_state, d, **kwargs)
                nengo.Connection(self.state_input, self.state, synapse=None)
                lmu_s = LMUProcess(theta=theta, q=q_n, size_in=n_neurons_state,with_resets=True)
                self.state_act_memory = nengo.Node(lmu_s)
                nengo.Connection(self.state.neurons, self.state_act_memory[1:],synapse=tau)
                nengo.Connection(self.reset, self.state_act_memory[0],synapse=None)
                
            
            # LMU to remember state 
            lmu_s = LMUProcess(theta=theta, q=q_s,size_in=d,with_resets=True)
            self.state_memory = nengo.Node(lmu_s)
            nengo.Connection(self.state, self.state_memory[1:], synapse=0.05)
            nengo.Connection(self.reset, self.state_memory[0], synapse=None)

            # Env reward, an input
            self.reward_input = nengo.Node(size_in=1)
            
            # Reward representation being learned: R, a vector of dim d
            self.reward_SP = nengo.Ensemble(n_neurons_r, d)
    
            # The estimate of the current reward (a scalar) is the dot product of the learned 
            # reward representation and the current state: R . s_t
            self.reward_estimate = nengo.Node(lambda t,x: np.sum(x[:d]*x[d:]), size_in=d*2,size_out=1)
            nengo.Connection(self.reward_SP,self.reward_estimate[0:d], synapse=0.05)
            nengo.Connection(self.state,self.reward_estimate[d:], synapse=0.05)
    
            # Reward TD error is   (R.dot(s_t) - r_t ) * s_t
            self.reward_error = nengo.Node(lambda t,x: x[0]*x[1:] if t<T_test else 0, size_in=d+1,size_out=d)
            nengo.Connection(self.reward_estimate,self.reward_error[0], synapse=None)
            nengo.Connection(self.reward_input,self.reward_error[0], transform=-1, synapse=None)
            nengo.Connection(self.state, self.reward_error[1:], synapse=0.05)
            # The learned reward rep can depend on the 'context', TODO: use state_memory_ens instead of state
            learn_connR = nengo.Connection(self.state.neurons, self.reward_SP, 
                                             transform= np.zeros((d,n_neurons_state)),
                                  learning_rule_type=nengo.PES(learning_rate=reward_learning_rate))
            nengo.Connection(self.reward_error, learn_connR.learning_rule, synapse=None)

            
            
             # SR representation being learned, M(s_t), dim d
            self.sr = nengo.Ensemble(n_neurons_sr,d)
            # LMU to remember the SR
            lmu_sr = LMUProcess(theta=theta, q=q_sr,size_in=d,with_resets=True)
            self.sr_memory = nengo.Node(lmu_sr)
            nengo.Connection(self.sr, self.sr_memory[1:], synapse=0.05)
            nengo.Connection(self.reset, self.sr_memory[0], synapse=None)
            
            learn_connSR = nengo.Connection(self.state.neurons, self.sr, transform=np.zeros((d,n_neurons_state)), 
                                   learning_rule_type=DPES(d,n_neurons_state,1,learning_rate=sr_learning_rate))

            self.sr_error = nengo.Node(lambda t,x: x if t<T_test else np.zeros(d+n_neurons_sr),
                                       size_in= d + n_neurons_sr)
            nengo.Connection(self.state_memory, self.sr_error[:d], 
                             transform= -state_decoders, synapse= None  )
            nengo.Connection(self.sr, self.sr_error[:d], 
                             transform= -sr_decoders, synapse=0.05)
            nengo.Connection(self.sr_memory, self.sr_error[:d], 
                             transform = sr_memory_decoders, synapse=None)
            nengo.Connection(self.state_act_memory, self.sr_error[d:], 
                             transform =state_act_decoders, synapse=None)
            nengo.Connection(self.sr_error,learn_connSR.learning_rule, synapse= None)

    
            # Can compute the value given the SR and reward function
            self.value = nengo.Node(lambda t,x: np.sum(x[:d]*x[d:]),size_in=d*2,size_out=1)
            nengo.Connection(self.reward_SP, self.value[:d], synapse= 0.05)
            nengo.Connection(self.sr, self.value[d:], synapse= 0.05)
    

            

#Function to return decoders need for LMU RL rules (TD0, TDtheta, TDlambda)
#( these are descibed below)

# The decoders returned are activity_lmu_transform, reward_lmu_transform, value_transform, value_lmu_transform
# 
# The DPES rule will recieve input pre_activity and error (in a single vector)
# The update to state decoders will be of the form,
#    pre_activity @ error.T
# pre_activity is computed as
#    activity_lmu_transform @ A
# where A is either decoded activities or an LMU representation of activites depending on the rule
# The error is computed with three terms,
#    value_lmu_transform @ V - value_transform @ v - reward_lmu_transform @ R
# where v is the current value of the function we are learning (value usually, but could be Q or SR)
# V is the LMU representation of that function
# The function we want to learn is a discounted sum/integral of some variable, usually reward
#  R is the LMU representation of the reward
# For value learning r is the reward
# For SR learning r is the env state
def get_critic_transforms(rule_type, discount, n_neurons, theta, size=1,
                    q_a=10, q_r = 10, q_v=10, alpha=10, lambd=0.8):
    
    if rule_type=="TD0":
        activity_lmu_transform = eval_sh_legendre(np.arange(q_a).reshape(1,-1), 0)
        activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform)

        reward_lmu_transform = eval_sh_legendre(np.arange(q_r).reshape(1,-1), 0 )
        
        value_transform = 0
        legs = lpn(q_v-1, 0) 
        value_lmu_transform = (np.log(discount)*legs[0] - (1/theta)*legs[1]).reshape(1,-1)
    elif rule_type=="TD0euler":
        activity_lmu_transform = np.asarray([legendre(i)(1) for i in range(q_a)])
        activity_lmu_transform = np.kron(np.eye(n_neurons), activity_lmu_transform.reshape(q_a, -1).T)
    
        reward_lmu_transform = np.asarray([legendre(i)(1) for i in range(q_r)])
        reward_lmu_transform = np.kron(np.eye(size), reward_lmu_transform.reshape(q_r, -1).T)
        
        value_transform = 0 #1/theta

        dt = 0.1*theta
        mat1 = np.asarray([legendre(i)(1) for i in range(q_v)])
        mat1 =np.kron(np.eye(size), mat1.reshape(q_v, -1).T)
        mat2 = np.asarray([legendre(i)(0.9) for i in range(q_v)])
        mat2 =np.kron(np.eye(size), mat2.reshape(q_v, -1).T)
        value_lmu_transform = (np.log(discount) - 1/dt)*mat1 + mat2/dt

        
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



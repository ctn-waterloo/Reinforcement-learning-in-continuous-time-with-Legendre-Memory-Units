import nengo
from nengo.network import Network
import numpy as np
import scipy.linalg
from scipy.special import legendre, eval_legendre, eval_sh_legendre
import scipy.integrate as integrate
from scipy.special import lpn

from nengo import PES
from lmu_networks import LMUProcess, LMUNetwork_v2, LMUModulatedNetwork_v2

def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))



class ValueCritic(nengo.Network):
    def __init__(self,n_neurons_state, n_neurons_value, theta, d, discount, q_s, q_r, q_v,
                 algor, learn_schedule,
                 learning_rate=1e-4, T_test=10000,tau=0.05, lambd=0.8, dt=0.001, replay_type="forward",#"backward", "shuffled"
                  **kwargs):
        super().__init__()
        if not hasattr(learn_schedule, "__len__"): # array or tuple: length of "gather exp time" and "learn time":
            learn_schedule = [learn_schedule,learn_schedule]
        else:
            assert len(learn_schedule)==2

        # taus = np.flip(np.linspace(1,0,int(learn_schedule[1]/dt),endpoint=False))
        self.state_lmu_transform, self.reward_lmu_transform, self.value_lmu_transform, _ = get_dyn_critic_transforms(algor, discount, theta, learn_schedule[1], d,#
                                                     q_s=q_s, q_r = q_r, q_v=q_v, lambd=lambd)

        with self:
            def learn_schedule_fun(t):
                n = t // (learn_schedule[0] + learn_schedule[1])
                if ((t - n*(learn_schedule[0] + learn_schedule[1])) // learn_schedule[0] ) > 0:
                    return 1 # learn on
                else:
                    return 0 # learn off
            self.learn_on = nengo.Node(lambda t: learn_schedule_fun(t) if t< T_test else 0)
            self.reset = nengo.Node(size_in=1)

            self.state_input = nengo.Node(lambda t,x: x if ((learn_schedule_fun(t)==0) | (t>=T_test)) else np.zeros(d), size_in=d)
            self.state = nengo.Ensemble(n_neurons_state, d, **kwargs)
            nengo.Connection(self.state_input, self.state, synapse=None)
            
            lmu_s = LMUProcess(theta=theta, q=q_s,size_in=d, with_holds=True, with_resets=True)
            self.state_memory = nengo.Node(lmu_s)
            nengo.Connection(self.state, self.state_memory[2:],synapse=tau)
            nengo.Connection(self.learn_on, self.state_memory[0],synapse=None) # hold memory during learning

            def lmu_state_learn(t,x):
                n = t // (learn_schedule[0] + learn_schedule[1])
                if ((t - n*(learn_schedule[0] + learn_schedule[1])) // learn_schedule[0] ) <= 0:
                    return np.zeros(d)
                elif t>=T_test:
                    return np.zeros(d)
                else:
                    if replay_type=="forward":
                        ttau = 1 - (t -n*(learn_schedule[0] + learn_schedule[1]) - learn_schedule[0] )/learn_schedule[1]
                    elif replay_type=="backward":
                        ttau = (t -n*(learn_schedule[0] + learn_schedule[1]) - learn_schedule[0] )/learn_schedule[1]
                    elif replay_type=="shuffled":
                        ttau = np.random.rand()
                    mat = self.state_lmu_transform(ttau)
                    return mat @ x#

            self.state_memory_input = nengo.Node(lmu_state_learn, size_in=q_s*d)
            nengo.Connection(self.state_memory, self.state_memory_input, synapse=None)
            nengo.Connection(self.state_memory_input, self.state, synapse=None)

            self.reward_input = nengo.Node(size_in=1)
            lmu_r = LMUProcess(theta=theta, q=q_r,size_in=1, with_holds=True, with_resets=True)
            self.reward_memory = nengo.Node(lmu_r)
            nengo.Connection(self.reward_input, self.reward_memory[2:], synapse=None) 
            nengo.Connection(self.learn_on, self.reward_memory[0],synapse=None) # hold memory during learning

            lmu_v = LMUProcess(theta=theta, q=q_v,size_in=1, with_holds=True, with_resets=True)
            self.value = nengo.Ensemble(n_neurons_value,1)
            self.value_memory = nengo.Node(lmu_v)
            nengo.Connection(self.value, self.value_memory[2:], synapse=tau)
            nengo.Connection(self.learn_on, self.value_memory[0],synapse=None) # hold memory during learning
            
            self.learn_connV = nengo.Connection(self.state.neurons, self.value, 
                                                transform=np.zeros((1,n_neurons_state)), 
                                        learning_rule_type = PES(learning_rate = learning_rate),synapse=tau)

            def lmu_error_fun(t,x):
                m_r = x[:q_r]
                m_v = x[q_r:]
                n = t // (learn_schedule[0] + learn_schedule[1])
                if ((t - n*(learn_schedule[0] + learn_schedule[1])) // learn_schedule[0] ) <= 0:
                    return 0
                elif t>=T_test:
                    return 0
                else:
                    if replay_type=="forward":
                        ttau = 1 - (t -n*(learn_schedule[0] + learn_schedule[1]) - learn_schedule[0] )/learn_schedule[1]
                    elif replay_type=="backward":
                        ttau = (t -n*(learn_schedule[0] + learn_schedule[1]) - learn_schedule[0] )/learn_schedule[1]
                    elif replay_type=="shuffled":
                        ttau = np.random.rand()
                    # ttau = 2*ttau - 1
                    r_mat = self.reward_lmu_transform(ttau)
                    v_mat = self.value_lmu_transform(ttau)
                    return  - ( r_mat @ m_r) -( v_mat @ m_v ) #+ self.value_transform*v
   
                
            
            self.error = nengo.Node(lmu_error_fun, size_in=q_r + q_v )
            
            nengo.Connection(self.reward_memory, self.error[:q_r], synapse=tau)
            nengo.Connection(self.value_memory, self.error[q_r:],synapse=tau)

            def lmu_r_fun(t,x):
                m_r = x
                n = t // (learn_schedule[0] + learn_schedule[1])
                if ((t - n*(learn_schedule[0] + learn_schedule[1])) // learn_schedule[0] ) <= 0:
                    return 0
                elif t>=T_test:
                    return 0
                else:
                    if replay_type=="forward":
                        ttau = 1 - (t -n*(learn_schedule[0] + learn_schedule[1]) - learn_schedule[0] )/learn_schedule[1]
                    elif replay_type=="backward":
                        ttau = (t -n*(learn_schedule[0] + learn_schedule[1]) - learn_schedule[0] )/learn_schedule[1]
                    elif replay_type=="shuffled":
                        ttau = np.random.rand()
                    r_mat = self.reward_lmu_transform(ttau)
                    return   r_mat @ m_r 

            self.recalled_r = nengo.Node(lmu_r_fun, size_in = q_r)
            nengo.Connection(self.reward_memory, self.recalled_r, synapse=tau)
            
            nengo.Connection(self.error, self.learn_connV.learning_rule, synapse=None)
            self.rule = self.learn_connV.learning_rule





def get_dyn_critic_transforms(rule_type, discount, theta, w, state_size=1, taus= None,
                    q_s=10, q_r = 10, q_v=10, alpha=10, lambd=0.8,  n_samples=20):
    
    if rule_type=="TD0":
        def state_lmu_transform(tau):
            mat = eval_sh_legendre(np.arange(q_s).reshape(1,-1), tau) #2*tau - 1
            mat = np.kron(np.eye(state_size), mat)
            return mat
        def reward_lmu_transform(tau):
            return eval_sh_legendre(np.arange(q_r).reshape(1,-1), tau )
        def value_lmu_transform(tau):
            legs = lpn(q_v-1, 2*tau-1) 
            return (np.log(discount)*legs[0] - (1/theta)*legs[1]).reshape(1,-1)
        if taus is not None:
            dt = taus[1]-taus[0]
            state_mats = np.zeros((len(taus), state_size, state_size*q_s))
            reward_mats = np.zeros((len(taus), 1, q_r))
            value_mats = np.zeros((len(taus), 1, q_v))
            for i,t in enumerate(taus):
                state_mats[i] = state_lmu_transform(t)
                reward_mats[i] = reward_lmu_transform(t)
                value_mats[i] = value_lmu_transform(t)
            def state_lmu_transform(tau):
                return state_mats[int(tau/dt - dt)]
            def reward_lmu_transform(tau):
                return reward_mats[int(tau/dt - dt)]
            def value_lmu_transform(tau):
                return value_mats[int(tau/dt - dt)]
    elif rule_type=="TDtheta":
        def state_lmu_transform(tau):
            mat = eval_sh_legendre(np.arange(q_s).reshape(1,-1), tau) #2*tau - 1
            mat = np.kron(np.eye(state_size), mat)
            return mat
        def reward_lmu_transform(tau):
            return ((tau/n_samples)*np.sum(eval_sh_legendre(np.arange(q_r).reshape(1,-1), np.linspace(0,tau,n_samples).reshape(-1,1)), axis=0 )).reshape(1,-1)
        def value_lmu_transform(tau):
            mat1 = discount**(tau*theta)*eval_sh_legendre(np.arange(q_v).reshape(1,-1),0 )
            mat2 = - eval_sh_legendre(np.arange(q_v).reshape(1,-1),tau )
            return mat1 +mat2
        if taus is not None:
            dt = taus[1]-taus[0]
            state_mats = np.zeros((len(taus), state_size, state_size*q_s))
            reward_mats = np.zeros((len(taus), 1, q_r))
            value_mats = np.zeros((len(taus), 1, q_v))
            for i,t in enumerate(taus):
                state_mats[i] = state_lmu_transform(t)
                reward_mats[i] = reward_lmu_transform(t)
                value_mats[i] = value_lmu_transform(t)
            def state_lmu_transform(tau):
                return state_mats[int(tau/dt - dt)]
            def reward_lmu_transform(tau):
                return reward_mats[int(tau/dt - dt)]
            def value_lmu_transform(tau):
                return value_mats[int(tau/dt - dt)]
    else:
        print("Not a valid rule")
    return state_lmu_transform, reward_lmu_transform, value_lmu_transform, 0




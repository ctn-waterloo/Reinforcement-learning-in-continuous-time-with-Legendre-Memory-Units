import nengo
import numpy as np
from nengo.network import Network
import warnings
import scipy

from nengo.builder import Builder, Signal
from nengo.builder.connection import get_eval_points, solve_for_decoders
from nengo.builder.operator import (
    DotInc, ElementwiseInc, Operator, Reset, SimPyFunc)
from nengo.exceptions import ValidationError
from nengo.learning_rules import LearningRuleType
from nengo.params import EnumParam, FunctionParam, NumberParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.params import Default
from nengo.ensemble import Ensemble, Neurons
from nengo.builder.connection import slice_signal
from nengo.node import Node
from nengo.exceptions import BuildError



######################################
#       Delayed PES rule: its pes but the error signal also includes delayed activities
#####################################


def build_or_passthrough(model, obj, signal):
    """Builds the obj on signal, or returns the signal if obj is None."""
    return signal if obj is None else model.build(obj, signal)

def get_post_ens(conn):
    """Get the output `.Ensemble` for connection."""
    return (
        conn.post_obj
        if isinstance(conn.post_obj, (Ensemble, Node))
        else conn.post_obj.ensemble
    )



class DPES(LearningRuleType):
    modifies = "decoders"
    probeable = ("error", "activities", "delta")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-4)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)

    def __init__(self, error_size, pre_n_neurons, q_pre_neurons=1, learning_rate=Default, pre_synapse=Default):
        super(DPES, self).__init__(learning_rate, size_in=(pre_n_neurons + error_size)*q_pre_neurons)
        self.pre_n_neurons = pre_n_neurons
        self.error_size = error_size
        self.q_pre_neurons = q_pre_neurons
        self.pre_synapse = pre_synapse



class SimDPES(Operator):
    def __init__(self, error_size, pre_n_neurons, q_pre_neurons, 
                 pre_filtered, error, delta, learning_rate, tag=None):
        super(SimDPES, self).__init__(tag=tag)

        self.learning_rate = learning_rate

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, error]
        self.updates = [delta]
        self.error_size = error_size
        self.pre_n_neurons = pre_n_neurons
        self.q_pre_neurons = q_pre_neurons

    @property
    def delta(self):
        return self.updates[0]

    @property
    def error(self):
        return self.reads[1]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def _descstr(self):
        return f"pre={self.pre_filtered}, error={self.error} -> {self.delta}"
    
    @property
    def pre(self):
        return self.reads[0]

    @property
    def decoders(self):
        return self.updates[0]


    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.error][self.error_size*self.q_pre_neurons:].reshape(self.pre_n_neurons,-1)
        #.reshape(-1,self.d)
        error = signals[self.error][:self.error_size*self.q_pre_neurons].reshape(self.error_size,-1)
        delta = signals[self.delta]
        n_neurons = pre_filtered.shape[0]
        alpha = -self.learning_rate * dt / n_neurons

        def step_simpes():
            #np.outer(alpha * error, pre_filtered, out=delta)
            #delta = alpha * error @ pre_filtered.T
            #delta = pre_filtered @ (alpha * error)
            delta[...] = alpha *error @ pre_filtered.T #np.tensordot(error, pre_filtered.T, axes=((), ()))
        return step_simpes

    

@Builder.register(DPES)
def build_dpes(model, dpes, rule):
    conn = rule.connection

    # Create input error signal
    error = Signal(shape=rule.size_in, name="DPES:error")
    model.add_op(Reset(error))
    model.sig[rule]["in"] = error  # error connection will attach here

    # Filter pre-synaptic activities with pre_synapse
    acts = build_or_passthrough(
        model,
        dpes.pre_synapse,
        slice_signal(
            model,
            model.sig[conn.pre_obj]["out"],
            conn.pre_slice,
        )
        if isinstance(conn.pre_obj, Neurons)
        else model.sig[conn.pre_obj]["out"],
    )

    if isinstance(conn.post_obj, Neurons):# or isinstance(conn.pre_obj, Ensemble):
        # multiply error by post encoders to get a per-neuron error
        #   i.e. local_error = dot(encoders, error)
        post = get_post_ens(conn)
        if not isinstance(conn.post_slice, slice):
            raise BuildError(
                "DPES learning rule does not support advanced indexing on non-decoded "
                "connections"
            )

        encoders = model.sig[post]["encoders"]
        # slice along neuron dimension if connecting to a neuron object, otherwise
        # slice along state dimension
        encoders = (
            encoders[:, conn.post_slice]
            if isinstance(conn.post_obj, Ensemble)
            else encoders[conn.post_slice, :]
        )

        local_error = Signal(shape=(encoders.shape[0],))
        model.add_op(Reset(local_error))
        model.add_op(DotInc(encoders, error, local_error, tag="DPES:encode"))
    else:
        local_error = error

    model.add_op(SimDPES(dpes.error_size, dpes.pre_n_neurons, dpes.q_pre_neurons, acts,
                         local_error, model.sig[rule]["delta"], dpes.learning_rate))

    # expose these for probes
    model.sig[rule]["error"] = error
    model.sig[rule]["activities"] = acts



#####################################
#Synaptic Modulation rule: multiples decoders by a modulator signal
########################################

class SynapticModulation(LearningRuleType):
    modifies = "decoders"
    probeable = ("modulation")

    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)

    def __init__(self,  pre_synapse=Default):
        super(SynapticModulation, self).__init__(learning_rate=0,size_in=1)
        self.pre_synapse = pre_synapse

class SimSynapticModulation(Operator):
    def __init__(self, modulation, delta, weights, tag=None):
        super().__init__(tag=tag)
        self.reads = [modulation, weights]
        self.updates = [delta]
        self.sets = []
        self.incs = []
        
    @property
    def delta(self):
        return self.updates[0]
    
    @property
    def modulation(self):
        return self.reads[0]
    
    @property
    def weights(self):
        return self.reads[1]
    
#     @property
#     def decoders(self):
#         return self.updates[0]

    def make_step(self, signals, dt, rng):
        weights = signals[self.weights]
        delta = signals[self.delta]
        rate = signals[self.modulation]
#         print(delta, rate)
        def step_simsynmod():
             delta[...] = (rate-1)*weights
#             printf("HELLO", flush=True)
        return step_simsynmod



@Builder.register(SynapticModulation)
def build_synapticmodulation(model, synmod, rule):
    conn = rule.connection

    # Create input modulation signal
    modulation = Signal(shape=rule.size_in, name="SynapticModulation:modulation")
    model.add_op(Reset(modulation))
    model.sig[rule]["in"] = modulation  # mod connection will attach here
    model.add_op(SimSynapticModulation(modulation, model.sig[rule]["delta"], model.sig[conn]["weights"]))
    # expose these for probes
    model.sig[rule]["modulation"] = modulation
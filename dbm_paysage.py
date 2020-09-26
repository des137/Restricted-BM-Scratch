Source: https://github.com/drckf/paysage/blob/master/paysage/models/dbm.py

import os, pandas, operator
from cytoolz import partial
from typing import List

class BoltzmannMachine(object):
    def __init__(self, layer_list: List, conn_list: List = None):
        self.layers = layer_list
        self.num_layers = len(self.layers)
        self.clamped_sampling = []
        self.multipliers = [None for _ in range(self.num_layers)]
        self.connections = conn_list if conn_list is not None else self._default_connections()
        self.count_connections()
        for layer in self.layers:
            layer.update_moments(
                layer.conditional_mean([be.zeros((1,1))], [be.zeros((1, layer.len))]))

    def count_connections(self):
        self.num_connections = len(self.connections)

    def _default_connections(self):
        conns = []
        for i in range(self.num_layers - 1):
            w = layers.Weights((self.layers[i].len, self.layers[i+1].len))
            conns.append(mg.Connection(i, i+1, w))
        return conns

    def num_parameters(self):
        c = 0
        for l in self.layers:
            c += l.num_parameters()
        for conn in self.connections:
            c += conn.weights.shape[0]*conn.weights.shape[1]
        return c

    def get_config(self) -> dict:
        config = {
            "type": "BoltzmannMachine",
            "layers": [layer.get_config() for layer in self.layers],
            "connections": [conn.get_config() for conn in self.connections]
            }
        return config

    @classmethod
    def from_config(cls, config: dict):
        layer_list = [layers.layer_from_config(l) for l in config["layers"]]
        conn_list = None
        if "connections" in config:
            conn_list = [mg.Connection.from_config(c) for c in config["connections"]]
        return cls(layer_list, conn_list)

    def save(self, store: pandas.HDFStore) -> None:
        config = self.get_config()
        store.put('model', pandas.DataFrame())
        store.get_storer('model').attrs.config = config
        for i in range(self.num_layers):
            key = os.path.join('layers', 'layers_'+str(i))
            self.layers[i].save_params(store, key)
        for i in range(self.num_connections):
            key = os.path.join('connections', 'weights_'+str(i))
            self.connections[i].weights.save_params(store, key)

    @classmethod
    def from_saved(cls, store: pandas.HDFStore) -> None:
        config = store.get_storer('model').attrs.config
        model = cls.from_config(config)
        for i in range(len(model.layers)):
            key = os.path.join('layers', 'layers_'+str(i))
            model.layers[i].load_params(store, key)
        for i in range(len(model.connections)):
            key = os.path.join('connections', 'weights_'+str(i))
            model.connections[i].weights.load_params(store, key)
        return model

    def copy(self):
        model = BoltzmannMachine.from_config(self.get_config())
        for i in range(model.num_connections):
            model.connections[i].weights.set_params(self.connections[i].weights.params)
        for i in range(model.num_layers):
            model.layers[i].set_params(self.layers[i].get_params())
        return model

    def copy_params(self, model):
        for i in range(self.num_connections):
            self.connections[i].weights.set_params(model.connections[i].weights.params)
        for i in range(self.num_layers):
            self.layers[i].set_params(model.layers[i].get_params())

    def set_clamped_sampling(self, clamped_sampling):
        self.clamped_sampling = list(clamped_sampling)

    def get_sampled(self):
        return [i for i in range(self.num_layers) if i not in self.clamped_sampling]

    def _connected_rescaled_units(self, i: int, state: ms.State) -> List:
        units = []
        for conn in self.connections:
            if i == conn.target_index:
                units += [be.maybe_a(self.multipliers[conn.domain_index],
                                     self.layers[conn.domain_index].rescale(
                        state[conn.domain_index]), operator.mul)]
            elif i == conn.domain_index :
                units += [self.layers[conn.target_index].rescale(
                        state[conn.target_index])]
        return units

    def _connected_weights(self, i: int) -> List:
        weights = []
        for conn in self.connections:
            if i == conn.target_index:
                weights += [conn.weights.W(trans=True)]
            elif i == conn.domain_index:
                weights += [conn.weights.W(trans=False)]
        return weights

    def initialize(self, batch, method: str='hinton', **kwargs) -> None:
        try:
            func = getattr(init, method)
        except AttributeError:
            print(method + ' is not a valid initialization method for latent models')
        func(batch, self, **kwargs)

        for l in self.layers:
            l.enforce_constraints()

        for l in range(1, len(self.layers)):
            lay = self.layers[l]
            n = lay.len
            lay.update_moments(
                lay.conditional_mean([be.zeros((1,1))], [be.zeros((1, n))]))

        for conn in self.connections:
            conn.weights.enforce_constraints()

    def _alternating_update_(self, func_name: str, state: ms.State, beta=None) -> None:
        (odd_layers, even_layers) = (range(1, self.num_layers, 2),
                                     range(0, self.num_layers, 2))
        layer_order = [i for i in list(odd_layers) + list(even_layers)
                       if i in self.get_sampled()]

        for i in layer_order:
            func = getattr(self.layers[i], func_name)
            state[i] = func(
                self._connected_rescaled_units(i, state),
                self._connected_weights(i),
                beta=beta)

    def markov_chain(self, n: int, state: ms.State, beta=None,
                     callbacks=None) -> ms.State:
        new_state = ms.State.from_state(state)
        for _ in range(n):
            self._alternating_update_('conditional_sample', new_state, beta = beta)
            if callbacks is not None:
                for func in callbacks:
                    func(new_state)
        return new_state

    def mean_field_iteration(self, n: int, state: ms.State, beta=None,
                             callbacks=None) -> ms.State:
        new_state = ms.State.from_state(state)
        for _ in range(n):
            self._alternating_update_('conditional_mean', new_state, beta=beta)
            if callbacks is not None:
                for func in callbacks:
                    func(new_state)
        return new_state

    def deterministic_iteration(self, n: int, state: ms.State, beta=None,
                                callbacks=None) -> ms.State:
        new_state = ms.State.from_state(state)
        for _ in range(n):
            self._alternating_update_('conditional_mode', new_state, beta=beta)
            if callbacks is not None:
                for func in callbacks:
                    func(new_state)
        return new_state

    def compute_reconstructions(self, visible, method='markov_chain'):
        data_state = ms.State.from_visible(visible, self)
        return getattr(self, method)(1, data_state)

    def exclusive_gradient_(self, grad, state, func, penalize=True,
                            weighting_function=be.do_nothing):
        for i in range(self.num_layers):
            deriv = self.layers[i].derivatives(
                state[i],
                self._connected_rescaled_units(i, state),
                self._connected_weights(i),
                penalize=penalize,
                weighting_function=weighting_function
                )
            grad.layers[i] = [be.mapzip(func, z[0], z[1])
            for z in zip(deriv, grad.layers[i])]
        for i in range(self.num_connections):
            target = self.connections[i].target_index
            domain = self.connections[i].domain_index
            deriv = self.connections[i].weights.derivatives(
                self.layers[target].rescale(state[target]),
                self.layers[domain].rescale(state[domain]),
                penalize=penalize,
                weighting_function=weighting_function
                )
            grad.weights[i] = [be.mapzip(func, z[0], z[1])
            for z in zip(deriv, grad.weights[i])]
        return grad

    def gradient(self, data_state, model_state, data_weighting_function=be.do_nothing,
                 model_weighting_function=be.do_nothing):
        grad = gu.zero_grad(self)
        self.exclusive_gradient_(grad, data_state, be.add, penalize=True,
                                 weighting_function=data_weighting_function)
        self.exclusive_gradient_(grad, model_state, be.subtract, penalize=False,
                                 weighting_function=model_weighting_function)
        return grad

    def parameter_update(self, deltas):
        for layer_index in range(self.num_layers):
            self.layers[layer_index].parameter_step(deltas.layers[layer_index])
        for conn_index in range(self.num_connections):
            self.connections[conn_index].weights.parameter_step(deltas.weights[conn_index])

    def joint_energy(self, state):
        energy = 0
        for layer_index in range(self.num_layers):
            energy += self.layers[layer_index].energy(state[layer_index])
        for conn_index in range(self.num_connections):
            target = self.connections[conn_index].target_index
            domain = self.connections[conn_index].domain_index
            energy += self.connections[conn_index].weights.energy(
                    self.layers[target].rescale(state[target]),
                    self.layers[domain].rescale(state[domain]))
        return energy

    def _connected_elements(self, i: int, lst : List) -> List:
        connections = [lst[conn.target_index] for conn in self.connections
                       if i == conn.domain_index]
        connections += [lst[conn.domain_index] for conn in self.connections
                        if i == conn.target_index]
        return connections

    def _connecting_transforms(self, i: int, lst : List) -> List:
        connections = [lst[j] for j in range(len(self.connections))
                       if i == self.connections[j].domain_index]
        connections += [be.transpose(lst[j]) for j in range(len(self.connections))
                        if i == self.connections[j].target_index]
        return connections

    def _get_rescaled_weights(self) -> List:
        rescaled_w = []
        for conn in self.connections:
            target_scale = self.layers[conn.target_index].reciprocal_scale()
            domain_scale = self.layers[conn.domain_index].reciprocal_scale()
            rescaled_w.append(be.multiply(be.multiply(
                    be.unsqueeze(target_scale, axis=1), conn.weights.W()),
                    be.unsqueeze(domain_scale, axis=0)))
        rescaled_w2 = [be.square(w) for w in rescaled_w]
        return (rescaled_w, rescaled_w2)

    def gibbs_free_energy(self, cumulants, rescaled_weight_cache=None):
        if rescaled_weight_cache is None:
            rescaled_weight_cache = self._get_rescaled_weights()

        total = 0
        for index in range(self.num_layers):
            lay = self.layers[index]
            total += lay.TAP_entropy(cumulants[index])

        for index in range(self.num_connections):
            w = rescaled_weight_cache[0][index]
            w2 = rescaled_weight_cache[1][index]
            total -= be.quadratic(cumulants[index].mean, cumulants[index+1].mean, w)
            total -= 0.5 * be.quadratic(cumulants[index].variance,
                           cumulants[index+1].variance, w2)

        return total

    def compute_StateTAP(self, use_GD=True, init_lr=0.1, tol=1e-7, max_iters=50, ratchet=True,
                         decrease_on_neg=0.9, mean_weight=0.9, mean_square_weight=0.999,
                         seed=None, rescaled_weight_cache=None):
        if use_GD:
            return self._compute_StateTAP_GD(init_lr, tol, max_iters, ratchet,
                                             decrease_on_neg,
                                             mean_weight, mean_square_weight,
                                             rescaled_weight_cache=rescaled_weight_cache,
                                             seed=seed)
        else:
            return self._compute_StateTAP_self_consistent(tol=tol, max_iters=max_iters,
                                                          rescaled_weight_cache=rescaled_weight_cache,
                                                          seed=seed)

    def _compute_StateTAP_GD(self, init_lr=0.1, tol=1e-7, max_iters=50, ratchet=True,
                             decrease_on_neg=0.9, mean_weight=0.9, mean_square_weight=0.999,
                             seed=None, rescaled_weight_cache=None):
        state = seed
        if seed is None:
            state = ms.StateTAP.from_model_rand(self)
        cumulants = state.cumulants
        if rescaled_weight_cache is None:
            rescaled_weight_cache = self._get_rescaled_weights()
        free_energy = self.gibbs_free_energy(cumulants, rescaled_weight_cache)
        lr = be.float_scalar(init_lr)
        lr_ = partial(be.tmul_, lr)
        beta_1 = be.float_scalar(mean_weight)
        beta_2 = be.float_scalar(mean_square_weight)
        bt_1_ = partial(be.mix_, beta_1)
        bt_2_ = partial(be.mix_, beta_2)
        comp_bt1 = partial(be.tmul, be.float_scalar(1.0/(1.0 - beta_1)))
        comp_bt2 = partial(be.tmul, be.float_scalar(1.0/(1.0 - beta_2)))
        mom = [lay.get_zero_magnetization() for lay in self.layers]
        var = [lay.get_zero_magnetization() for lay in self.layers]
        var_corr = [lay.get_zero_magnetization() for lay in self.layers]
        grad = [lay.get_zero_magnetization() for lay in self.layers]
        eps = [be.apply(be.ones_like, mag) for mag in grad]
        for mag in eps:
            be.apply_(partial(be.tmul_, be.float_scalar(1e-6)), mag)
        coeff = [lay.get_zero_magnetization() for lay in self.layers]
        depth = range(self.num_layers)
        for _ in range(max_iters):
            new_grad = self._TAP_magnetization_grad(cumulants, rescaled_weight_cache)
            for i in depth:
                be.mapzip_(bt_1_, mom[i], new_grad[i])
                grad[i] = be.apply(comp_bt1, mom[i])
            if mean_square_weight > 1e-6:
                for i in depth:
                    be.mapzip_(bt_2_, var[i], be.apply(be.square, new_grad[i]))
                    var_corr[i] = be.apply(comp_bt2, var[i])
                for i in depth:
                    coeff[i] = be.apply(be.reciprocal,
                               be.mapzip(be.add, be.apply(be.sqrt, var_corr[i]), eps[i]))
                for c in coeff:
                    be.apply_(lr_,c)
                for i in depth:
                    grad[i] = be.mapzip(be.multiply, coeff[i], grad[i])
            else:
                for g in grad:
                    be.apply_(lr_,g)
            new_cumulants = [
                self.layers[l].clip_magnetization(
                    be.mapzip(be.subtract, grad[l], cumulants[l])
                )
                for l in range(self.num_layers)]
            new_free_energy = self.gibbs_free_energy(new_cumulants,
                                                     rescaled_weight_cache)
            neg = free_energy - new_free_energy < 0
            if abs(free_energy - new_free_energy) < tol:
                break
            if neg:
                lr *= decrease_on_neg
                lr_ = partial(be.tmul_, be.float_scalar(lr))
                if lr < 1e-10:
                    break
                if ratchet == False:
                    cumulants = new_cumulants
                    free_energy = new_free_energy
            else:
                cumulants = new_cumulants
                free_energy = new_free_energy

        return ms.StateTAP(cumulants, self.lagrange_multipliers_analytic(cumulants)), \
               free_energy

    def _compute_StateTAP_self_consistent(self, tol=1e-7, max_iters=50,
                                          seed=None, rescaled_weight_cache=None):
        state = seed
        if seed is None:
            state = ms.StateTAP.from_model_rand(self)
        if rescaled_weight_cache is None:
            rescaled_weight_cache = self._get_rescaled_weights()
        free_energy = self.gibbs_free_energy(state.cumulants, rescaled_weight_cache)
        for itr in range(max_iters):
            for i in range(self.num_layers-1, -1, -1):
                self.layers[i].update_lagrange_multipliers_(
                    state.cumulants[i],
                    state.lagrange_multipliers[i],
                    self._connected_elements(i, state.cumulants),
                    self._connecting_transforms(i, rescaled_weight_cache[0]),
                    self._connecting_transforms(i, rescaled_weight_cache[1]))
                self.layers[i].self_consistent_update_(
                    state.cumulants[i],
                    state.lagrange_multipliers[i])
            new_free_energy = self.gibbs_free_energy(state.cumulants, rescaled_weight_cache)
            if abs(free_energy - new_free_energy) < tol:
                break
            free_energy = new_free_energy
        return state, free_energy

    def lagrange_multipliers_analytic(self, cumulants):
        return [self.layers[i].lagrange_multipliers_analytic(cumulants[i])
                for i in range(self.num_layers)]

    def _TAP_magnetization_grad(self, cumulants, rescaled_weight_cache=None):
        if rescaled_weight_cache is None:
            rescaled_weight_cache = self._get_rescaled_weights()

        grad = [None for lay in self.layers]
        for i in range(self.num_layers):
            grad[i] = self.layers[i].TAP_magnetization_grad(
                cumulants[i],
                self._connected_elements(i, cumulants),
                self._connecting_transforms(i, rescaled_weight_cache[0]),
                self._connecting_transforms(i, rescaled_weight_cache[1]))
        return grad

    def _grad_gibbs_free_energy(self, state, rescaled_weight_cache=None):
        if rescaled_weight_cache is None:
            rescaled_weight_cache = self._get_rescaled_weights()
        grad_GFE = gu.Gradient(
            [self.layers[i].GFE_derivatives(state.cumulants[i],
                self._connected_elements(i, state.cumulants),
                self._connecting_transforms(i, rescaled_weight_cache[0]),
                self._connecting_transforms(i, rescaled_weight_cache[1]))
             for i in range(self.num_layers)]
            ,
            [conn.weights.GFE_derivatives(
                self.layers[conn.target_index].rescale_cumulants(
                    state.cumulants[conn.target_index]),
                self.layers[conn.domain_index].rescale_cumulants(
                    state.cumulants[conn.domain_index]),
                )
            for conn in self.connections]
            )

        return grad_GFE

    def grad_TAP_free_energy(self, use_GD=True, init_lr=0.1, tol=1e-7, max_iters=50, ratchet=True,
                             decrease_on_neg=0.9, mean_weight=0.9, mean_square_weight=0.999):
        rescaled_weight_cache = self._get_rescaled_weights()
        state,_ = self.compute_StateTAP(use_GD, init_lr, tol, max_iters, ratchet,
                                        decrease_on_neg, mean_weight, mean_square_weight,
                                        rescaled_weight_cache = rescaled_weight_cache)
        return self._grad_gibbs_free_energy(state,
                                            rescaled_weight_cache = rescaled_weight_cache)

    def TAP_gradient(self, data_state, use_GD=True, init_lr=0.1, tol=1e-7, max_iters=50,
                     ratchet=True, decrease_on_neg=0.9, mean_weight=0.9,
                     mean_square_weight=0.999):
        pos_phase = gu.null_grad(self)
        for i in range(self.num_layers):
            pos_phase.layers[i] = self.layers[i].derivatives(
                data_state[i],
                self._connected_rescaled_units(i, data_state),
                self._connected_weights(i),
                penalize=True)
        for i in range(self.num_connections):
            target = self.connections[i].target_index
            domain = self.connections[i].domain_index
            pos_phase.weights[i] = self.connections[i].weights.derivatives(
                self.layers[target].rescale(data_state[target]),
                self.layers[domain].rescale(data_state[domain]),
                penalize=True)
        neg_phase = self.grad_TAP_free_energy(use_GD, init_lr, tol, max_iters,
                                              ratchet, decrease_on_neg,
                                              mean_weight, mean_square_weight)

        grad = gu.grad_mapzip(be.subtract, neg_phase, pos_phase)
        return grad

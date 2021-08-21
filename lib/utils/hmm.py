import numpy as np
from scipy.stats import invgamma


class GaussianHmm(object):
    def __init__(self, frame_dim, n_states=3):

        self.n_states = n_states
        self.frame_dim = frame_dim

        # Params init
        self.initial_state_prob, self.a_trans = init_transition_matrices(n_states)

        # shape (n_states, frame_dim)
        self.mean_emissions = init_emission_means(n_states, frame_dim)
        # shape (n_states, frame_dim, frame_dim)
        self.cov_emissions = init_diagonal_cov_matrices(n_states, frame_dim)

    def sample_sequences(self, seq_length, n_sequences):
        # this data size is convenient for other libs like pytorch
        sequences = np.zeros((seq_length, n_sequences, self.frame_dim))

        # one initial hidden state per sequence
        hidden_states = np.random.choice(self.n_states, size=n_sequences,
                                         p=self.initial_state_prob)
        for idx_seq in range(seq_length):
            hidden_states = self.next_hidden_states(hidden_states)
            sequences[idx_seq] = self.emit_frame(hidden_states)
        return sequences

    def next_hidden_states(self, hidden_states_prev):
        # hidden_states_prev is a 1D array
        hidden_next = np.zeros(hidden_states_prev.size,
                               dtype=hidden_states_prev.dtype)
        for idx_seq, state_prev in enumerate(hidden_states_prev):
            hidden_next[idx_seq] = np.random.choice(self.n_states,
                                                    p=self.a_trans[state_prev])
        return hidden_next

    def emit_frame(self, hidden_states):
        n_sequences = hidden_states.size
        frame = np.zeros((n_sequences, self.frame_dim))
        for idx_seq, h_state in enumerate(hidden_states):
            mean = self.mean_emissions[h_state]
            cov = self.cov_emissions[h_state]
            frame[idx_seq] = np.random.multivariate_normal(mean, cov)
        return frame
    
    def emission_expected_value(self, frame_instant):
        prob_state_t = self.initial_state_prob @ np.linalg.matrix_power(
            self.a_trans, frame_instant+1)  # array of shape n_states
        
        # expected value = sum_{i=1}^N {\mu_i * P(S_t=i)}
        # \mu_i is the gauss mean of size frame_dim and N is the num of states
        expected_value = np.sum(
            self.mean_emissions * prob_state_t.reshape(self.n_states, 1), 
            axis=0)
        return expected_value
        

def init_transition_matrices(n_states):
    dirichlet_params = np.random.uniform(size=n_states)
    # matrix of shape (n_states, n_states) with probabilities in each row
    a_trans = np.random.dirichlet(
        alpha=dirichlet_params,
        size=n_states
    )
    init_state_prob = np.random.dirichlet(alpha=dirichlet_params)
    return init_state_prob, a_trans


def init_emission_means(n_states, frame_dim):
    emission_means = np.zeros((n_states, frame_dim))
    locs = np.random.uniform(low=-5, high=5, size=n_states)
    for i in range(n_states):
        emission_means[i] = np.random.normal(loc=locs[i], size=frame_dim)
    return emission_means


def init_diagonal_cov_matrices(n_states, frame_dim):
    cov_matrix = np.zeros((n_states, frame_dim, frame_dim))
    inv_gamma_params = np.random.uniform(low=1, high=5 , size=n_states)
    for i_state, gamma_param in enumerate(inv_gamma_params):
        diagonal = np.abs(invgamma.rvs(gamma_param, size=frame_dim))
        cov_matrix[i_state] = np.diag(diagonal)
    return cov_matrix

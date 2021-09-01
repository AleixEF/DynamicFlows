import numpy as np
from scipy.stats import invgamma, multivariate_normal


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

        # one hidden state per sequence at t=0
        hidden_states = np.random.choice(self.n_states, size=n_sequences,
                                         p=self.initial_state_prob)
        for frame_instant in range(seq_length):
            sequences[frame_instant] = self.emit_frame(hidden_states)
            hidden_states = self.next_hidden_states(hidden_states)
        return sequences

    def loglike_sequence(self, sequence):
        # initial calculation at t=0
        # shape (n_states, batch_size)
        frame_prob = self.compute_emission_prob(sequence[0])

        # shape (n_states, batch_size)
        alpha_temp = frame_prob * self.initial_state_prob.reshape((self.n_states, 1))
        c_coef = np.sum(alpha_temp, axis=0)  # shape batch_size
        alpha_hat = alpha_temp / c_coef  # shape (n_states, batch_size)

        logprob = np.log(c_coef)
        # starting loop from t=1
        for frame in sequence[1:]:
            frame_prob = self.compute_emission_prob(frame)
            alpha_temp = (self.a_trans.T @ alpha_hat) * frame_prob
            c_coef = np.sum(alpha_temp, axis=0)
            alpha_hat = alpha_temp / c_coef

            logprob += np.log(c_coef)
        return logprob
    
    def save(self, folder_path):
        np.save(folder_path+"/init_prob.npy", self.initial_state_prob)
        np.save(folder_path+"/a_trans.npy", self.a_trans)
        np.save(folder_path+"/covariances.npy", self.cov_emissions)
        np.save(folder_path+"/mean_emissions.npy", self.mean_emissions)
        return

    def load(self, folder_path):
        self.initial_state_prob = np.load(folder_path+"/init_prob.npy")
        self.a_trans = np.load(folder_path+"/a_trans.npy")
        self.cov_emissions = np.load(folder_path+"/covariances.npy")                                     
        self.mean_emissions = np.load(folder_path+"/mean_emissions.npy")
        return self

    def emission_expected_value(self, frame_instant):
        prob_state_t = self.initial_state_prob @ np.linalg.matrix_power(
            self.a_trans, frame_instant)  # array of shape n_states
        # expected value = sum_{i=1}^N {\mu_i * P(S_t=i)}
        # \mu_i is the gauss mean of size frame_dim and N is the num of states
        expected_value = np.sum(
            self.mean_emissions * prob_state_t.reshape(self.n_states, 1),
            axis=0)
        return expected_value

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

    def compute_emission_prob(self, frame_batch):
        batch_size, frame_dim = frame_batch.shape
        prob_per_hidden_state = np.zeros((self.n_states, batch_size))
        for i in range(self.n_states):
            prob_per_hidden_state[i] = multivariate_normal.pdf(
                    x=frame_batch,
                    mean=self.mean_emissions[i],
                    cov=self.cov_emissions[i])
        return prob_per_hidden_state


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

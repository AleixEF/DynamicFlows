import torch
from sklearn.metrics import classification_report
import numpy as np

from context import hmm, flows, esn


def train(nf, esn_model, optimizer, sequences_batch):
    loglike = nf.loglike_sequence(sequences_batch, esn_model)
    loss = -torch.mean(loglike)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def predict(models_list, esn_model, single_sequence):
    n_categories = len(models_list)
    likelihoods = torch.zeros(n_categories)
    for cat in range(n_categories):
        likelihoods[cat] = models_list[cat].loglike_sequence(single_sequence, esn_model)
    return torch.argmax(likelihoods)


def main():
    n_categories = 5
    n_updates = 100

    seq_length = 10
    batch_size = 64
    frame_dim = 34
    hidden_layer_dim = 34

    n_flow_layers = 4
    learning_rate = 1e-5

    esn_model = esn.EchoStateNetwork(frame_dim)
    flow_models = [flows.NormalizingFlow(frame_dim, hidden_layer_dim, num_flow_layers=n_flow_layers)
                   for _ in range(n_categories)]  # one model per category

    optimizers = [torch.optim.SGD(nf.parameters(), lr=learning_rate) for nf in flow_models]
    # one hmm per category to generate sequences
    data_gens = [hmm.GaussianHmm(frame_dim) for _ in range(n_categories)]

    for nf in flow_models:
        nf.train()

    for update_idx in range(n_updates):

        # tr_loss_running = 0.0
        model_category = 0

        for nf, optim, data_gen in zip(flow_models, optimizers, data_gens):
            sequence_batch = data_gen.sample_sequences(seq_length, batch_size)
            sequence_batch = torch.from_numpy(sequence_batch).float()

            loss = train(nf, esn_model, optim, sequence_batch)

            if update_idx % 10 == 0:
                print("Update no. {}, loss for model {}: {}".format(update_idx,
                                                                    model_category,
                                                                    loss.item()))
                model_category += 1

    for nf in flow_models:
        nf.eval()

    n_test_sequences = 100
    true_categories = np.random.randint(low=0, high=n_categories, size=n_test_sequences)
    predictions = np.zeros(n_test_sequences, dtype=np.int32)  # to fill

    with torch.no_grad():
        for i, true_cat in enumerate(true_categories):
            single_sequence = data_gens[true_cat].sample_sequences(seq_length, n_sequences=1)
            single_sequence = torch.from_numpy(single_sequence).float()
            predictions[i] = predict(flow_models, esn_model, single_sequence)

        print(classification_report(true_categories, predictions))

    return None


if __name__ == '__main__':
    main()

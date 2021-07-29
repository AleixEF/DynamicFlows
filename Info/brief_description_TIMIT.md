# The TIMIT Dataset -  A brief description

The TIMIT dataset [[1]](https://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database)[[2]](https://ieeexplore.ieee.org/iel1/29/1758/00046546.pdf?casa_token=sXvGQS2ptokAAAAA:rMibo9lch5X03TUxV5seLNWkugLwYZ6RKKqrGU-HfLfXfCbXclIpYc0lHnJnCl8UB3dOYktL), which has sentences manually labelled at the phoneme level. Sentences are in spoken English. In total, TIMIT consists of 6300 sentences, where 10 sentences are spoken by 630 speakers. The original TIMIT transcriptions consists of 61 speech phones. 

## Specific details:
- The speech signal in the dataset has been sampled at 16 kHz,
- The dataset consists of 6300 phoneme-level speech utterances that have been split into two sets:
    - A training set consisting of 4620 utterances and,
    - A testing set consisting of 1680 utterances. 
- In the original TIMIT dataset there are 61 phones available for classification. There also exists a smaller, 'folded' set of 39 phones mapped from the larger set of phones [[2]](https://ieeexplore.ieee.org/iel1/29/1758/00046546.pdf?casa_token=sXvGQS2ptokAAAAA:rMibo9lch5X03TUxV5seLNWkugLwYZ6RKKqrGU-HfLfXfCbXclIpYc0lHnJnCl8UB3dOYktL). Models are usually evaluated for classification performance on the folded set of 39 phones. 

## Speech phones: Brief introduction
TODO

## Other relevant details
TODO

# Dataset for noise addition
Varieties of additive noises from from the NOISEX-92 database (also sampled at the same frequency of 16 kHz) can be considered. Snippets can be taken from random offsets in the noise database and can be added to the clean test data at different signal-to-noise ratio (SNR) levels (with respect to the clean test signal) to generate noisy training / test sets. 

# References

[[1]](https://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database) Lopes, Carla, and Fernando Perdigao. "Phone recognition on the TIMIT database." Speech Technologies/Book 1 (2011): 285-302.

[[2]](https://ieeexplore.ieee.org/iel1/29/1758/00046546.pdf?casa_token=sXvGQS2ptokAAAAA:rMibo9lch5X03TUxV5seLNWkugLwYZ6RKKqrGU-HfLfXfCbXclIpYc0lHnJnCl8UB3dOYktL) Lee, K-F., and H-W. Hon. "Speaker-independent phone recognition using hidden Markov models." IEEE Transactions on Acoustics, Speech, and Signal Processing 37.11 (1989): 1641-1648.
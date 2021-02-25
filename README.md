# Master-s-thesis
Master's thesis regarding semi-supervised learning for automatic speech recognition (ASR). 

Two teacher/student (T/S) networks were implemented to explore semi-supervised learning for ASR. The teacher model was built on a deep neural network (DNN), 
and the student model was built on the same DNN architecture. The T/S networks were trained on Mel-frequency cepstral coefficients (MFCC) and
feature- space maximum likelihood linear regression (fMLLR) features. Both monophones and triphones were used as targets in two separate networks. 
The features and the targets were obtained through the Kaldi toolkit.

The experiments conducted in this paper were evaluated on the TIMIT Corpus data set. The TIMIT corpus has a training set, 
a validation set, and a test set. The training set has 3696 utterances, the validation has 400 utterances, and the test set has 192 utterances.


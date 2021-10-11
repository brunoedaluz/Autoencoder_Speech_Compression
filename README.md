# Convolutional Autoencoder for Speech Compression

Autoencoders, as defined by [1], are neural networks with a central layer smaller than the inputs that have the property to perform dimensionality reduction and also reconstruct the input vectors, with the capacity to do more powerful generalizations than the PCA algorithm.
The autoencoders have applications in image compression tasks [2,3,4], where some algorithms compete with the most used JPEG. Recent researches have been showed the use of deep autoencoders to compress speech spectrograms [5] and general audio compression [6].
This work proposes and implements an end-to-end speech compression model based on a Convolutional Autoencoder, realizes experiments in speech compression and decompression and shows reconstruction errors, compression rate and quality perception based on PESQ [7] metric.

# Related Work

Deep Learning is reaching great success on computer vision tasks and there are some good examples on image compression. The work of [4] achieves a significant result compressing images compared to traditional algorithms like JPEG2000, using a Deep Convolutional Autoencoder.
In [5] is presented a Convolutional Variational Autoencoder (Convolutional VAE) for spectrogram compression. According to the authors, spectrograms as features show better results than the use of Mel-frequency Cepstral Coefficients (MFCC) for training Automatic Speech Recognition (ASR) systems. The work of [6] proposes a combination between Recurent  Neural Networks (RNN) and Variational Autoencoders (VA) to the task of end-to-end audio compression, with the possibility of separation between the Encoder and Decoder, which is necessary for compression tasks. This research relies on the success of the previous compressing images works and on the capacity to extend the image pattern recognition well performed by convolutional neural networks to audio spectrograms.

# Proposed Model

The proposed model is an end-to-end process to compress and decompress speech audio. The audio inputs are converted to spectrograms through STFT and the Neural Network is trained with a training dataset. After that, the model can receive an audio input, extract its spectrogram, and compress it. The compression and decompression tasks are presentedin Figure 1. It’s main module is an Autoencoder Convolutional Neural Network based on the model created by [8]. The encoder is composed by 3 convolutional layers and 3 MaxPooling layers stacked, with the respective sizes adjusted to the input sizes. The decoder has 4 convolutional layers and 3 UpSampling layers stacked. The overall network is represented on Figure2.  The activation function used in the convolutional layers isthe Rectified Linear Unit (ReLU).The  training  data  is  composed  by  speech  spectrogramscreated through Short-Time Fourier Transform (STFT) fromLibrispeech dataset audio files.The encoder output is the latent representation of the inputspectrograms, or the compressed data. The decoder takes thiscompressed data and decompress it to a spectrogram.

# Experiment

The proposed model is implemented and tested in 5 phases, represented in Figure 3.

Phase 1 - Data Preparation

In this phase, the data is loaded in the execution environmentand transformed through 6 steps.

1. Audio data load - the dataset used is Librispeech [9], specifically the "development set, clean speech", composed by 2703 "flac" format audio files, with 16kHz sample rate. The data is loaded with Librosa library functions, resulting in 891 audio samples. The test dataset used is the Librispeech "test set, clean speech", composed by 859 audio samples.

2. STFT and spectrogram - in this step, the STFT is calculated for each training and test dataset sample. For the training dataset, is stored the absolute value of STFT, i.e, its spectrogram. The spectrogram is a 257 x 1001 size matrix. To facilitate the calculations, one line and one column from the spectrogram is trimmed so the result matrix is multiple by 2. The final matrix size is 256x 1000.

3. Scaling - the spectrogram matrices are scaled using the scikit-learn function "StandardScaler", that subtracts each value by the mean and divides by the variance. This makes the values more suitable to be used as neural network inputs. This procedure is performed separately in the training and test datasets.

4. Reshaping - each 2 dimension spectrogram matrix is reshaped in a 3 dimensional matrix, size 256 x 1000 x 1, which is the shape required by the convolutional layers of the network.

Phase 2 - Convolutional Autoencoder Training

The model presented is implemented in Python, using the Tensorflow-Keras libraries, with the layers sizes, inputs and outputs sizes observerd in Figure  2. The training dataset is used as input and output of the network, splitted in 85% for training and 15% for validation. Are configured 500 epochs for the training, with early stop configured observing validation loss change.

Phase 3 - Convolutional Autoencoder Test

After the model training, the test dataset is used as autoencoder input and the predicted outputs are collected. These values are used to calculate the MAE (mean average error), that is the metric used to estimate the reconstruction error.

Phase 4 - Audio Reconstruction

The predicted outputs obtained in Phase 3 are used to reconstruct the audios. They are passed through the inverse process of the Phase 1, respectively, reshaping to 2 dimensions, inverse scaling, inverse STFT using the Fast Griffin-Lin method [10], which results in reconstructed audio data.

Phase 5 - Reconstructed Audio Evaluation

The reconstructed audio data produced in Phase 4 are compared to the original audio data using the PESQ metric [7].

# Results

The execution of the experiment produced a trained model with 4385 parameters in 274 epochs. The following metrics are calculated: compression rate, reconstruction error and PESQ metric.

Compression Rate

The compression rate can be calculated as the size relation between the spectrogram input and the latent layer representation of the network. The input data size is 256000 (256 x1000 matrix) and the encoder output (latent representation) is 32 x 125 x 8 (32000 total), which results in a compression rate of 8:1.

Reconstruction Error

The autoencoder reconstruction error is the difference between the input data and the output data, calculated as the MAE (Mean Average Error). For each input and output sample is calculated the difference element wise, after that, the average is the value for each sample. At the end, the mean for all samples are calculated. The value obtained on the test dataset is 0.417, which means a reconstruction error of 41.7% between the input and output.

PESQ Metric

While the reconstruction error gives an analytical measure between the input and output error values, the PESQ metric offers a more sophisticated evaluation, in quality perception levels. The PESQ values can be mapped in a scale from 1 (bad) to 5 (excellent) [11]. The PESQ value is calculated for each sample of the test dataset and the final value is the mean of all samples. The obtained value is 2.08.

# CONCLUSION AND FUTURE WORK

The model was implemented and tested successfully and the results were collected. Due to resources constraints, the training dataset and the Convolutional Autoencoder sizes  were limited. The results suggest that this approach is valid for the proposed task, because obtained a good compression rateof 8:1. However, the reconstruction error is high yet (41.7%), which reflected in the PESQ metric of 2.08, which falls in the poor category. The high reconstruction error can be result of a shallow size of the Neural Network and the data overfit.
To address these points, the following recommendations are made:

- Increase the training dataset size. Librispeech offers a 100 hour dataset consisting in more than 28000 samples. This large dataset can increase the model's accuracy, reduce the reconstruction error and reduce the overfit.
- Increase  the  Convolutional  Autoencoder  layers  quantity by adding more convolutional layers. This can increase the latent representation precision and reduce the reconstruction error.


# References

[1] G. E. Hinton, “Reducing the dimensionality of data with neural networks,” Science, vol. 313, no. 5786, pp. 504–507, July 2006.

[2] Johannes Ball ́e, Valero Laparra, and Eero P. Simoncelli, “End-to-end optimized image compression,” CoRR, vol.abs/1611.01704, 2016.

[3] Lucas Theis, Wenzhe Shi, Andrew Cunningham, and Ferenc Huszar, “Lossy image compression with com-pressive autoencoders,” 2017.

[4] Zhengxue Cheng, Heming Sun, Masaru Takeuchi, and Jiro  Katto, “Deep  convolutional AutoEncoder-based lossy image compression,” in2018 Picture Coding Symposium (PCS). June 2018, IEEE.

[5] Olga Yakovenko and Ivan Bondarenko, “Convolutional variational autoencoders for spectrogram compressionin automatic speech recognition,” in Communications in Computer and Information Science, pp. 115–126. Springer International Publishing, 2021.

[6] Daniela N. Rim, Inseon Jang, and Heeyoul Choi, “Deepneural networks and end-to-end learning for audio com-pression,” CoRR, vol. abs/2105.11681, 2021

[7] A.W. Rix, J.G. Beerends, M.P. Hollier, and A.P. Hekstra, “Perceptual evaluation of speech quality (PESQ) -a newmethod for speech quality assessment of telephone networks and codecs,” in 2001 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings (Cat. No.01CH37221). IEEE.

[8] Francois Chollet, “Building autoencoders in keras” 2016, Last accessed 13 September 2021.

[9] Vassil Panayotov, Guoguo Chen, Daniel Povey, and San-jeev Khudanpur, “Librispeech:  An ASR corpus based on public domain audio books,” in 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). Apr. 2015, IEEE.

[10] Nathanael Perraudin, Peter Balazs, and Peter L. Sonder-gaard,  “A fast griffin-lim algorithm,” in 2013 IEEEWorkshop on Applications of Signal Processing to Audio and Acoustics. Oct. 2013, IEEE

[11] ITU International Telecomunications Union ,“Mapping function for transforming raw result scores to moslqo,” 2003, Last accessed 11 September 2021.

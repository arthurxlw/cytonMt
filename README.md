# cytonMt

CytonMT: an Efficient Neural Machine Translation Open-source Toolkit Implemented in C++

Xiaolin Wang (xiaolin.wang@nict.go.jp, arthur.xlw@gmail.com)

================================================

To start using the toolkit 

(Note that you can run bash compile_train_translate.sh)

1) compiling the toolkit;

requirement: cuda, cudnn (>= 7.0)

  make -j8;

2) training a model;

  cd data

  ../bin/cytonMt --mode train --epochs 5  --train train.sn:train.tn --vocab vocab --dev dev.sn:dev.tn --embedSize 64 --hiddenSize 128 --numLayers 1 --saveModel model

3) translating a few sentences.

  ../bin/cytonMt --mode translate --loadModel model/model --testInput test.sn --testOutput test.trans 

(Note that the output will be junk for this example)

================================================

More examples:

1) To replicate our experiment on WMT 2014 English-to-German (please get the necessary files from running the command of "datagen --problems=translate_ende_wmt_bpe32k" in the tensor2tensor toolkit (https://github.com/tensorflow/tensor2tensor)

  ../bin/cytonMT --mode train --probeFreq 12 --patience 12 --train train.tok.clean.bpe.32000.en:train.tok.clean.bpe.32000.de --vocab vocabFile --dev newstest2013.tok.bpe.32000.en:newstest2013.tok.bpe.32000.de  --saveModel model 

  ../bin/cytonMt --mode translate  --maxSeqLen 300 --loadModel model/model --testInput newstest2014.tok.bpe.32000.en --testOutput trans

2) To replicate our experiments on WMT 2017 English-to-German (please get the necessary files from running the scripts in https://github.com/marian-nmt/marian-examples/tree/master/wmt2017-uedin)

  ../bin/cytonMT --mode train --probeFreq 36 --patience 12 --train corpus.bpe.en:corpus.bpe.de:1.0:news.2016.bpe.en:news.2016.bpe.de:0.5 --vocab vocabFile --dev valid.bpe.en:valid.bpe.de --saveModel model 

  ../bin/cytonMt --mode translate --maxSeqLen 300 --loadModel model/model --testInput test2017.bpe.en --testOutput trans 

3) To train a model in a fast way like OpenNMT

  ../bin/cytonMt --mode train --epochs 13 --decayStart 10 --decayConti 1 --train train.sn:train.tn --vocab train.sn.vocab:train.tn.vocab --dev dev.sn:dev.tn --saveModel model

================================================

If you are using our toolkit, please kindly cite our paper:

    @article{wang2018cytonmt,
      title={CytonMT: an Efficient Neural Machine Translation Open-source Toolkit Implemented in C++},
      author={Wang, Xiaolin and Utiyama, Masao and Sumita, Eiichiro},
      journal={arXiv preprint arXiv:1802.07170},
      year={2018}
    }


================================================

The parameters of CytonMT

bin/cytonMt --help
  version: 2018-0528
  --help	 ()
  --mode	train/translate ()
  --saveModel	 ()
  --loadModel	load model for continue training or translate ()
  --maxSaveModels	maximum number of saved models (10)
  --train	source-side and target-side training files, one sentences per line. trainSrc:trainTrg[:weight:trainSrc2:trainSrc2:weight2] (trainSrc:trainTrg)
  --dev	source-side and target-side development files, one sentences per line (devSrc:devTrg)
  --testInput	input file for translating (testInput)
  --testOutput	output file for translating (testOutput)
  --vocab	source-side and target-side vocabulary files, one word per line (vocabSrc:vocabTrg)
  --srcTrgShareEmbed	share the embedding weight between the source side and the target side (1)
  --srcVocabSize	size of source-side vocabulary, 0 means using whole vocabulary in vocabSrc file (0)
  --trgVocabSize	size of source-side vocabulary, 0 means using whole vocabulary in vocabTrg file (0)
  --ignoreUnk	0/1, 1 means ignoring unknown words (1)
  --initParam	initialize weights uniformly in (-initParam, initParam) (0.1)
  --optimization	SGD/Adam (SGD)
  --learningRate	learning rate (1)
  --decayRate	decay factor of learning rate (0.7)
  --decayStart	learning rate start to decay from the epoch of decayStart (1000)
  --decayConti	0/1, 1 means that learning rate keeps decaying per check once it decays, OpenNMT's mode,   (0)
  --decayStatus	0/1, 1 means that learning rate is in a status of decaying, useful for continue training. (0)
  --epochs	max epochs of training (100)
  --epochStart	the number of first epoch, useful for continue training (1)
  --batchSize	batch size (64)
  --maxSeqLen	max length of source and target sentence (100)
  --embedSize	size of word embedding (512)
  --hiddenSize	size of hidden states (512)
  --numLayers	number of encoder/decoder layers (2)
  --dropout	dropout rate, 0 means disabling dropout (0.2)
  --clipGradient	threshold for clip gradient (5)
  --labelSmooth	factor of smoothing the target labels (0.1)
  --probeFreq	number of times probing the development likelihood per epoch (1)
  --probeMargin	margin for checking whether the development likelihood has increased (0.01)
  --patience	threshold for decaying the learning rate and restart training from the best model (1)
  --beamSize	size of beam search in translating (10)
  --lenPenalty	length penalty (0.6)






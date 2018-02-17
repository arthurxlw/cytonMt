#1) compiling the toolkit;
make -j8;

#2) training a model;
cd data
../bin/cytonMt --mode train --epochs 5  --decayRate 0.99 --train train.sn:train.tn --vocab train.sn.vocab:train.tn.vocab --dev dev.sn:dev.tn --saveModel model

#3) translating a few sentences.
../bin/cytonMt --mode translate --loadModel model/model --testInput test.sn --testOutput test.trans 

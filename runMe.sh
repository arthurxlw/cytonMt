#1) compiling the toolkit;
make -j8;

#2) training a model;
cd data
../bin/cytonMt --mode train --epochs 5  --train train.sn:train.tn --vocab vocab --dev dev.sn:dev.tn --embedSize 64 --hiddenSize 128 --numLayers 1 --saveModel model

#3) translating a few sentences.
../bin/cytonMt --mode translate --loadModel model/model --testInput test.sn --testOutput test.trans 

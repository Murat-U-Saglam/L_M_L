kaggle competitions download -c dogs-vs-cats
mkdir data
unzip dogs-vs-cats.zip -d data
rm dogs-vs-cats.zip
rm data/sampleSubmission.csv
unzip data/train.zip -d data
unzip data/test1.zip -d data

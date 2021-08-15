
Before run this model:

1.use this repo (https://github.com/yhcc/OntoNotes-5.0-NER) to preprocess ontonotes 5.0 dataset and drop the output in folder “/arabic” like:

    /arabic/dev.txt
    /arabic/test.txt
    /arabic/train.txt

2.run processing.py to process the output data into bert-tokenized format, the result will be saved into "bert_tokenized_dev.txt","bert_tokenized_train.txt","bert_tokenized_test.txt"
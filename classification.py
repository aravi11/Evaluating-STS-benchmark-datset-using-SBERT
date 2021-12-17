from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datetime import datetime
import sys, os, logging
import math, random
import gzip, csv

# Model and dataset paths 
model_name = 'roberta-base'
nli_dataset_path = './AllNLI.tsv.gz'
sts_dataset_path = './stsbenchmark.tsv.gz'
model_save_path = 'classification_output/v4_training_nli_'+model_name.replace("/", "-")

#Training parameters
train_batch_size = 128          
max_seq_length = 50
num_epochs = 4


word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


print('Starting to read training dataset for classification...')

def add_to_samples(sent1, sent2, label):
    if sent1 not in train_data:
        train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
    train_data[sent1][label].add(sent2)


train_data = {}
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'train':
            sent1 = row['sentence1'].strip()
            sent2 = row['sentence2'].strip()

            add_to_samples(sent1, sent2, row['label'])
            add_to_samples(sent2, sent1, row['label'])  


train_samples = []
for sent1, others in train_data.items():
    if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
        train_samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
        train_samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))

print("Train samples: {}".format(len(train_samples)))


print('Removing data duplicates within a batch...')
train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)


# Training loss function
train_loss = losses.MultipleNegativesRankingLoss(model)

#Read STSbenchmark dataset and use it as development set
print("Reading STSbenchmark dev dataset for fine tuning...")
dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')


warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
print("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=int(len(train_dataloader)*0.5),
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=False          #Set to True, if your GPU supports FP16 operations
          )


print('Model has been trained and fine-tuned successfully. Storing it in - '+ model_save_path)

test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'test':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

#Load the saved model
model = SentenceTransformer(model_save_path)

print('Evaluating the STS Bechmark test dataset with the trained model....')
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)
 # Learning sentence representations from natural language inference data   
 
## Description   

The first practical of the Advanced Topics in Computational Semantics course concerns learning
general-purpose sentence representations in the natural language inference (NLI) task. The goal
of this practical is threefold:
• to implement four neural models to classify sentence pairs based on their relation;
• to train these models using the Stanford Natural Language Inference (SNLI) corpus (Bowman
et al., 2015);
• and to evaluate the trained models using the SentEval framework (Conneau and Kiela, 2018).
NLI is the task of classifying entailment or contradiction relationships between premises and
hypotheses, such as the following:
Premise Bob is in his room, but because of the thunder and lightning outside, he cannot sleep.
Hypothesis 1 Bob is awake.
Hypothesis 2 It is sunny outside.
Hypothesis 3 Bob is lying in his bed.
While the first hypothesis follows from the premise, indicated by the alignment of ‘cannot sleep’
and ‘awake’, the second hypothesis contradicts the premise, as can be seen from the alignment of
‘sunny’ and ‘thunder and lightning’ and recognizing their incompatibility. The third hypothesis is
not necessarily entailed by the premise, and neither is contradicted. Therefore, its relation to the
premise is considered to be neutral.
For a model to recognize textual entailments, it has to reason about semantic relationships
within sentences. Hence, a thorough understanding of natural language is required which can be
transferred to other tasks involving natural language. In this assignment, we focus on pretraining
a sentence encoder on NLI, and afterwards evaluate its sentence embeddings on a variety of natural
language tasks.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   

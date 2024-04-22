# Report

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

## Data

Table: Model Vs Accuracy

| Model              | val_acc | test_acc |
|--------------------|---------|----------|
| Avg               | .72     | .72      |
| LSTM               | .80     | .80      |
| BiLSTM             | .80     | .79      |
| BiLSTM-Max-Pooling | .84     | .84      |


Table: Model Vs Sentval Tasks

| Model             | MR    | CR    | MPQA  | SUBJ  | SST2  | TREC | MRPC  | SICKEntailment |
|-------------------|-------|-------|-------|-------|-------|------|-------|----------------|
| Avg               | 76.84 | 46.54 | 78.06 | 90.73 | 79.24 | 83.0 | 73.58 | 81.0           |
| LSTM              | 72.9  | 75.87 | 86.98 | 85.76 | 77.64 | 73.2 | 72.82 | 84.96          |
| BiLSTM            | 75.18 | 77.85 | 87.11 | 88.96 | 80.07 | 86.8 | 72.4  | 84.6           |
| BiLSTM-MaxPooling | 77.01 | 79.07 | 88.2  | 91.32 | 83.03 | 89.2 | 73.9  | 83.8           |


## Analysis


We compute the following sentiment in the analysis.ipynb

Premise - “Two men sitting in the sun”
Hypothesis - “Nobody is sitting in the shade”
Label - Neutral (likely predicts contradiction)

Premise - “A man is walking a dog”
Hypothesis - “No cat is outside”
Label - Neutral (likely predicts contradiction)

We observed all of trained models predict contradiction. The reason for the failure can be attributed to
Premise - “Two men sitting in the sun”
Hypothesis - “Nobody is sitting in the shade”
can be the fact that the model thinks two men and nobody contradict each other. Hence it thinks the two sentence are contradiction. And also train data lacks world knowlege about sun and shade. It doesn't understand there is arelationship between sun and shade. Hence it might treated both sun and shade the same. Hence resulting the contradiction.


For the following sentence, 
Premise - “A man is walking a dog”
Hypothesis - “No cat is outside”
is hard to understand for a model. Because the model needs to understand the presence of a man and a dog doesn't provide any information about cat. The model doesn't understand that type of logic. Hence, it thinks that if a dog is outside and then everything should be outside including a cat. Thus, if cat is not outside, then it predicts a contradiction.

I suspect we can improve the accuracy by having dataset that's has more varied logical structures. Then the model might understand the suble logical difference as show above.





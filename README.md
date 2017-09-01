# kaggle-quora-nlp
The trainng dataset consists of **149263** pairs of questions that have similar meaning and **255025** pairs of questions that are dissimilar.

## PV-DM & PV-BOW methods
### Training on unprocessed data
| Iter | PV-DM | PV-BOW |
|------|-------|--------|
|  10  | 343s  |  523s  |
|  25  | 865s  |  831s  |

- Checking top3 inferred vector similarity scores for test vector tag```41```<br>
 doc2vec_dm: ('252202', 0.572), ('388290', 0.563), ('289333', 0.548)<br>
 
- Checking top3 inferred vector similarity scores for test vector tag```41```<br>
doc2vec_bow: ('41', 0.836), ('68375', 0.739), ('162563', 0.708)

- TODO:
1. Use varying param values.

### Training on processed data
- TODO

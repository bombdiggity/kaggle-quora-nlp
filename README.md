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
| Iter | PV-DM | PV-BOW |
|------|-------|--------|
|  10  | 797s  |  317s  |
|  25  | 806s  |  766s  |

##### Model Params:
| size | window | min count | Iter |
|------|--------|-----------|------|
| 100  | 5      | 5         |  25  |


- doc2vec_dm: Top3 inferred vector similarity scores for test vector tag```41 [u'rockets',u'look',u'white'] ```<br>

|   Tag  | Score | Text                                          |
|--------|-------|-----------------------------------------------|
| 289333 | 0.662 | pakistani people look white                   |
| 41     | 0.607 | rockets look white                            |
| 430016 | 0.571 | africanamerican christians think jesus white  |
 
 
- doc2vec_bow: Top3 inferred vector similarity scores for test vector tag```41 [u'rockets',u'look',u'white'] ```<br>

|   Tag  | Score | Text                                          |
|--------|-------|-----------------------------------------------|
| 41     | 0.771 | rockets look white                            |
| 53956  | 0.743 | photon look like                              |
| 72398  | 0.724 | black hole look like                          |

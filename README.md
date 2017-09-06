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


- <b>doc2vec_dm</b>: Top3 inferred vector similarity scores for test vector tag```41 [u'rockets',u'look',u'white'] ```<br>

|   Tag  | Score | Text                                                         |         
|--------|-------|--------------------------------------------------------------|
| 192611 | 0.618 | suitable inpatient drug alcohol rehab center white county ar |
| 478558 | 0.610 | pays transaction charges white lebel ams                     |
| 289333 | 0.609 | pakistani people look white                                  |

The cosine similarity score for ```41 [u'rockets',u'look',u'white'] ``` using this model is <b>0.597</b>. Per the above model, <b>Tag 41</b> is the <b>6th</b> closest compared to all other tags.
 
 
- <b>doc2vec_bow</b>: Top3 inferred vector similarity scores for test vector tag```41 [u'rockets',u'look',u'white'] ```<br>

|   Tag  | Score | Text                                          |
|--------|-------|-----------------------------------------------|
| 41     | 0.771 | rockets look white                            |
| 53956  | 0.743 | photon look like                              |
| 72398  | 0.724 | black hole look like                          |

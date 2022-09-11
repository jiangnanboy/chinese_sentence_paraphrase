## chinese_sentence_paraphrase

pytorch code for Integrating Linguistic Knowledge to Sentence Paraphrase Generation(AAAI2020 paper).

chinese blog [结合语言知识的句子改写生成](https://www.cnblogs.com/little-horse/p/15815537.html).

### Data Preprocessing
data from 'data/sentence_polishing/sentence_polishing_data.rar'

* STEP 1. build vocab
```
python data_processing.py
```

* STEP 2. build paraphrase pairs
```
python prepro_dict.py
```

### Train
chinese pretrained word embedding from [sgns.sogou.word](https://github.com/Embedding/Chinese-Word-Vectors).
```
nohup python -u train.py > log.log 2>&1 &
```

### evaluation
```
this code in train.py
```

### inference
```
this code in train.py
```

Refer to `hparams.py` for more parameters details.


## Reference
- https://github.com/bentrevett/pytorch-seq2seq
- https://github.com/LINMouMouZiBo/KEPN
- Integrating Linguistic Knowledge to Sentence Paraphrase Generation(Zibo Lin, Ziran Li, Ning Ding, Hai-Tao Zheng, Ying Shen, Wei Wang and Cong-Zhi Zhao. AAAI2020 paper)
- https://github.com/Embedding/Chinese-Word-Vectors

## contact

如有搜索、推荐、nlp以及大数据挖掘等问题或合作，可联系我：

1、我的github项目介绍：https://github.com/jiangnanboy

2、我的博客园技术博客：https://www.cnblogs.com/little-horse/

3、我的QQ号:2229029156
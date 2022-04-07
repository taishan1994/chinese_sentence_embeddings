# chinese_sentence_embeddings
bert_avg，bert_whitening，sbert，consert，simcse，esimcse 中文句向量表示

主要是跑了以下[zhoujx4/NLP-Series-sentence-embeddings: NLP句子编码、句子embedding、语义相似度：BERT_avg、BERT_whitening、SBERT、SmiCSE (github.com)](https://github.com/zhoujx4/NLP-Series-sentence-embeddings)的代码。使用的预训练的模型是：hfl_chinese-roberta-wwm-ext。其中sup_simcse（SNLI）需要先运行data_utils.py生成SNLI处理后的数据。原始的数据可以去：[https://github.com/pluto-junzeng/CNSD](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fpluto-junzeng%2FCNSD)下载。

```
model   dev test
bert_avg    Pearson: 0.254920    Spearman: 0.205940

bert_whitening  Pearson: 0.758313    Spearman: 0.688894

sbert
2022-04-06 17:43:18 - Cosine-Similarity :	Pearson: 0.8251	Spearman: 0.8257
2022-04-06 17:43:18 - Manhattan-Distance:	Pearson: 0.7989	Spearman: 0.8093
2022-04-06 17:43:18 - Euclidean-Distance:	Pearson: 0.7980	Spearman: 0.8084
2022-04-06 17:43:18 - Dot-Product-Similarity:	Pearson: 0.7874	Spearman: 0.7956

2022-04-06 17:43:25 - Cosine-Similarity :	Pearson: 0.7919	Spearman: 0.7837
2022-04-06 17:43:25 - Manhattan-Distance:	Pearson: 0.7706	Spearman: 0.7694
2022-04-06 17:43:25 - Euclidean-Distance:	Pearson: 0.7705	Spearman: 0.7690
2022-04-06 17:43:25 - Dot-Product-Similarity:	Pearson: 0.7550	Spearman: 0.7536

unsup_consert:
2022-04-06 17:58:23 - Cosine-Similarity :	Pearson: 0.7788	Spearman: 0.7808
2022-04-06 17:58:23 - Manhattan-Distance:	Pearson: 0.7497	Spearman: 0.7684
2022-04-06 17:58:23 - Euclidean-Distance:	Pearson: 0.7503	Spearman: 0.7691
2022-04-06 17:58:23 - Dot-Product-Similarity:	Pearson: 0.7572	Spearman: 0.7629

2022-04-06 17:58:27 - Cosine-Similarity :	Pearson: 0.7325	Spearman: 0.7241
2022-04-06 17:58:27 - Manhattan-Distance:	Pearson: 0.7126	Spearman: 0.7118
2022-04-06 17:58:27 - Euclidean-Distance:	Pearson: 0.7128	Spearman: 0.7116
2022-04-06 17:58:27 - Dot-Product-Similarity:	Pearson: 0.7182	Spearman: 0.7086

unsup_simcse:
2022-04-06 18:28:59 - Cosine-Similarity :	Pearson: 0.7835	Spearman: 0.7867
2022-04-06 18:28:59 - Manhattan-Distance:	Pearson: 0.7715	Spearman: 0.7875
2022-04-06 18:28:59 - Euclidean-Distance:	Pearson: 0.7717	Spearman: 0.7875
2022-04-06 18:28:59 - Dot-Product-Similarity:	Pearson: 0.7783	Spearman: 0.7822

2022-04-06 18:29:03 - Cosine-Similarity :	Pearson: 0.7539	Spearman: 0.7454
2022-04-06 18:29:03 - Manhattan-Distance:	Pearson: 0.7414	Spearman: 0.7424
2022-04-06 18:29:03 - Euclidean-Distance:	Pearson: 0.7420	Spearman: 0.7432
2022-04-06 18:29:03 - Dot-Product-Similarity:	Pearson: 0.7550	Spearman: 0.7450

sup_simcse:
2022-04-07 13:58:43 - Cosine-Similarity :	Pearson: 0.6292	Spearman: 0.6529
2022-04-07 13:58:43 - Manhattan-Distance:	Pearson: 0.6544	Spearman: 0.6555
2022-04-07 13:58:43 - Euclidean-Distance:	Pearson: 0.6514	Spearman: 0.6531
2022-04-07 13:58:43 - Dot-Product-Similarity:	Pearson: 0.6299	Spearman: 0.6548

2022-04-07 13:58:47 - Cosine-Similarity :	Pearson: 0.6400	Spearman: 0.6550
2022-04-07 13:58:47 - Manhattan-Distance:	Pearson: 0.6684	Spearman: 0.6584
2022-04-07 13:58:47 - Euclidean-Distance:	Pearson: 0.6631	Spearman: 0.6542
2022-04-07 13:58:47 - Dot-Product-Similarity:	Pearson: 0.6351	Spearman: 0.6398

unsup_esimcse:
2022-04-07 10:18:45 - Cosine-Similarity :	Pearson: 0.7881	Spearman: 0.7901
2022-04-07 10:18:45 - Manhattan-Distance:	Pearson: 0.7738	Spearman: 0.7912
2022-04-07 10:18:45 - Euclidean-Distance:	Pearson: 0.7743	Spearman: 0.7921
2022-04-07 10:18:45 - Dot-Product-Similarity:	Pearson: 0.7822	Spearman: 0.7854

2022-04-07 10:18:49 - Cosine-Similarity :	Pearson: 0.7467	Spearman: 0.7393
2022-04-07 10:18:49 - Manhattan-Distance:	Pearson: 0.7324	Spearman: 0.7382
2022-04-07 10:18:49 - Euclidean-Distance:	Pearson: 0.7322	Spearman: 0.7384
2022-04-07 10:18:49 - Dot-Product-Similarity:	Pearson: 0.7452	Spearman: 0.7379
```

| 模型               | Chinese-STS-B-dev | Chinese-STS-B-test | 训练参数                                                     |
| ------------------ | ----------------- | ------------------ | ------------------------------------------------------------ |
| bert_avg           | 0.2549            | 0.2059             | batch_size=32. max_len=64, pooling=cls                       |
| bert_whitening     | 0.7583            | 0.6888             | /                                                            |
| sbert              | 0.8257            | 0.7837             | batch_size=32. max_len=64, epoch=2, lr=2e-5                  |
| unsup_consert      | 0.7808            | 0.7241             | batch_size=32. max_len=64, epoch=2, lr=2e-5                  |
| unsup_simcse       | 0.7867            | 0.7454             | batch_size=32. max_len=64, epoch=2, lr=2e-5                  |
| sup_simcse（SNLI） | 0.6529            | 0.6550             | batch_size=32. max_len=64, epoch=1, lr=2e-5 <br />（感觉这个有问题） |
| unsup_esimcse      | 0.7901            | 0.7393             | batch_size=32. max_len=64, epoch=4, lr=2e-5                  |

还可以去参考：

[NLP-model/model/model/Torch_model/SimCSE-Chinese at main · zhengyanzhao1997/NLP-model (github.com)](https://github.com/zhengyanzhao1997/NLP-model/tree/main/model/model/Torch_model/SimCSE-Chinese)<.>

[vdogmcgee/SimCSE-Chinese-Pytorch: SimCSE在中文上的复现，有监督+无监督 (github.com)](https://github.com/vdogmcgee/SimCSE-Chinese-Pytorch)<br>

[KwangKa/SIMCSE_unsup: 中文无监督SimCSE Pytorch实现 (github.com)](https://github.com/KwangKa/SIMCSE_unsup)<br>

发现其实代码都是差不多的= =

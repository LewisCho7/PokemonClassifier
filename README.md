# PokemonClassifier
Classify pokemon using CNN

Used transfer learning by selecting backbone model ResNet18 and ResNet34.
Dataset: https://www.kaggle.com/datasets/lantian773030/pokemonclassification/data
Checked the impact of pretrained weights and fine-tuning range on model performance.
5 epochs trained and evaluated

### 5 experiments

| Experiment | Model | Pretrained Weights | Fine-tuning Range |
| :--- | :--- | :---: | :--- |
| **Exp 1** | ResNet18 | Yes | FC Layer Only |
| **Exp 2** | ResNet18 | Yes | Full Layers |
| **Exp 3** | ResNet18 | No | Full Layers |
| **Exp 4** | ResNet18 | Yes | Partial Layers |
| **Exp 5** | ResNet34 | Yes | FC Layer Only |

### Evaluation results(5 epoch)
| Experiment | Accuracy | Precision | Recall |
| :--- | :---: | :---: | :---: |
| Exp1 | 92.67% | 0.9399 | 0.9269 |
| Exp2 | 93.70% | 0.9544 | 0.9385 |
| Exp3 | 43.18% | 0.4619 | 0.4284 |
| Exp4 | 97.95% | 0.9789 | 0.9770 |
| Exp5 | 94.13% | 0.9459 | 0.9417 |

### Learning Curve
<img width="1000" height="600" alt="learning_curves" src="https://github.com/user-attachments/assets/ea09d83e-2067-49d3-9b73-9bdaba1c7c27" />

### Analysis
- Exp1 used the pretrained weights of ResNet18 and fine-tuning range was only the FC layer. Other layers were fixed.
- Compared with Exp1, Exp2 fine-tuned full layers which lead to higher accuracy, precision and recall. This shows that fine-tuning has impact on the performance. However it takes more time to train all the layers
- Compared with Exp1 and 2, Exp3 has significantly low accuracy. This shows the importance of pretrained weights from the model. Lack of prior knowledge from the pre-trained model ResNet leads to poor performance. However, as number of epoch increases and more data is provided, the accuracy will eventually increase.
- Compared with Exp1 and 2, Exp4 has better accuracy, precision and recall. This shows that the partial layer was trained to effectively distinguish the features of pokemon.
- Compare with the rest of the experiments, exp5 used different backbone of ResNet34. It has better accuracy than exp1 even though they both fine-tuned the FC layer only. This means that the increased depth of the model had impact on performance.
- When looking at the learning curve, Exp4 and Exp2 have low loss. This means that fine-tuning was very effective in catching the detailed features of pokemon. Exp3 has high loss because the pretrained weights were not used. It is interesting to see that exp1 and exp5 has similar loss and shape even though ResNet34 has more depth in layer. Both feature extraction method is effective in this pokemon classification. Maybe using ResNet18 is enough for this task.

### Simple app of classifing pokemon using the model exp4
Streamlit used
app.py -> streamlit run PATH
<img width="760" height="855" alt="img1" src="https://github.com/user-attachments/assets/a3d6d819-7f5d-454a-b2a5-bdeeb45e7c27" />
<img width="751" height="847" alt="img2" src="https://github.com/user-attachments/assets/0f764050-872b-4b6f-84f7-300b44e4e623" />
<img width="792" height="888" alt="img3" src="https://github.com/user-attachments/assets/4c2bcb89-3863-40e2-b294-f6c9799f1b57" />





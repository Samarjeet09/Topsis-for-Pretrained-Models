# Text Classification Model Evaluation Using Topsis

This project involves the evaluation of five different text classification models using the Hugging Face Transformers library and datasets. The models are pretrained on various architectures such as BERT, DistilBERT, RoBERTa, ALBERT, and XLNet. The evaluation is performed on a text classification dataset from Hugging Face datasets hub (IMDb dataset is used as an example).

## Data Generation

### Data Generation File: `comparison-of-text-classification-models.ipynb`

The data generation process involves importing libraries, defining models and datasets, preparing the dataset, and evaluating each model. The script uses the Hugging Face datasets library to load the IMDb dataset and then evaluates different text classification models on a subset of the data. The results are stored in a CSV file named `InputForTopsis.csv`.

## TOPSIS Analysis
### TOPSIS File: `topsis.ipynb`
The TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) analysis is performed on the evaluation results. The script reads the InputForTopsis.csv file, applies the TOPSIS algorithm, and generates an output CSV file named Output.csv. The TOPSIS score is calculated based on specified weights and impact factors.

## Evaluation Metrics

The `Output.csv` file contains the following evaluation metrics for each text classification model:

1. **Accuracy:**
   - Definition: The ratio of correctly predicted instances to the total instances.
   - Interpretation: A higher accuracy indicates better model performance.

2. **Precision:**
   - Definition: The ratio of correctly predicted positive observations to the total predicted positives.
   - Interpretation: High precision means a low false positive rate.

3. **Recall:**
   - Definition: The ratio of correctly predicted positive observations to the all observations in the actual class.
   - Interpretation: High recall means a low false negative rate.

4. **F1 Score:**
   - Definition: The weighted average of Precision and Recall.
   - Interpretation: F1 Score considers both false positives and false negatives.

5. **ROC-AUC Score:**
   - Definition: The Area Under the Receiver Operating Characteristic (ROC) curve.
   - Interpretation: ROC-AUC measures the model's ability to distinguish between classes.

6. **Average Precision:**
   - Definition: The area under the Precision-Recall curve.
   - Interpretation: High average precision indicates better performance in imbalanced datasets.

7. **Matthews Correlation Coefficient:**
   - Definition: A measure of the quality of binary classifications.
   - Interpretation: Values range from -1 to 1, with 1 being a perfect prediction.

8. **Cohen's Kappa:**
   - Definition: A statistic that measures inter-rater agreement.
   - Interpretation: Cohen's Kappa adjusts for the possibility of chance agreement.

9. **Log Loss:**
   - Definition: The logarithm of the likelihood function for a classification problem.
   - Interpretation: Lower log loss values indicate better model performance.

10. **TOPSIS Score:**
    - Definition: The TOPSIS score calculated using the Technique for Order of Preference by Similarity to Ideal Solution.
    - Interpretation: A higher TOPSIS score implies a better overall performance considering multiple criteria.

## Result and Analysis
`output.csv`
|FIELD1|Model                                    |Accuracy          |Precision         |Recall            |F1 Score          |ROC-AUC           |Average Precision |Matthews Correlation Coefficient|Cohen's Kappa     |Time (s)          |Log Loss          |score                |rank|
|------|-----------------------------------------|------------------|------------------|------------------|------------------|------------------|------------------|--------------------------------|------------------|------------------|------------------|---------------------|----|
|0     |mnoukhov/gpt2-imdb-sentiment-classifier  |0.9283333333333332|0.929406850459482 |0.9283333333333332|0.9282885136543674|0.9283333333333332|0.8888994708994709|0.8577395120046135              |0.8566666666666667|230.23796033859253|2.845196881538593 |0.29404885283435006  |4   |
|1     |XSY/albert-base-v2-imdb-calssification   |0.9325            |0.933657624937681 |0.9325            |0.9324549230423914|0.9325            |0.8941164817749603|0.8661568513509478              |0.865             |233.54750680923465|2.302352583806997 |0.7935750687882731   |1   |
|2     |wrmurray/roberta-base-finetuned-imdb     |0.945             |0.946433547725474 |0.945             |0.9449558117488762|0.945             |0.9099053627760252|0.8914323950537941              |0.89              |213.93476223945615|3.0957985114210183|0.26435664026121963  |5   |
|3     |lvwerra/distilbert-imdb                  |0.925             |0.9262123373150296|0.925             |0.924946628713752 |0.925             |0.8839794303797469|0.851211473980203               |0.85              |111.30703377723694|2.201056980993954 |0.7508109006647132   |2   |
|4     |JiaqiLee/imdb-finetuned-bert-base-uncased|0.9241666666666668|0.924904347826087 |0.9241666666666668|0.9241337386018236|0.9241666666666666|0.8848039999999999|0.8490706940404084              |0.8483333333333334|220.45876574516296|3.126624279306181 |0.0033577307551204832|6   |
|5     |Intradiction/text_classification_NoLORA  |0.9275            |0.9282434782608696|0.9275            |0.9274685193226226|0.9275            |0.8891960000000001|0.8557431552902348              |0.855             |110.99619698524477|2.43848710790093  |0.6584945155677805   |3   |

![full comp](https://github.com/Samarjeet09/Topsis-for-Pretrained-Models/blob/main/Images/CommpAll.png)
## Conclusion 
<img src="https://github.com/Samarjeet09/Topsis-for-Pretrained-Models/blob/main/Images/Score_comparison.png" width="800" height="800">
```For our limited usecase secnario we can conlcude that ALBERT model provides us with the best results ```


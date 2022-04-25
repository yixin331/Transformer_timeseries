## Enhanced Transformer model for time-series  

Much of the code is adapted from **Chart-to-Text: Generating Natural Language Explanations for Charts by Adapting the Transformer Model** (https://github.com/JasonObeid/Chart2Text)

### Project Description:
This study focuses on generating captions for time-series line charts.
We extend a time-series line chart dataset through crowdsourcing and adapt it to the enhanced Transformer model.

### Step0: Generating chart images for collections
Generate chart images for the new time-series dataset for caption collection: https://github.com/yixin331/ChartCaptioningSystem
Final_Dataset_new.xlsx contains the latest 100 time series while Final_Dataset.xslx contains the old 100 time series.

```
python Datasets/0_Monthly_single_timeseries_plot_generator.py
python Datasets/1_XLSX_to_JSON.py
python Datasets/2_Time_Series_Correlation_Analysis.py
```

### Step1: Retrieving and cleaning dataset

Retrieve the dataset:

```
python Datasets/0_Captions_retrieval.py
python Datasets/1_Captions_preprocessing.py
```
After that, manually combine the v1_captions_collections.xlsx and v1_new_captions_collections.xlsx to preprocessed_new_captions_collection.xlsx.

Clean the text within the chart titles and summaries:
```
python Datasets/cleanDataset.py
python utils/refactorTitles.py
python utils/refactorCaptions.py
```

### Step2: Preprocessing

```
python etc/templatePreprocess.py
```

* Converts data tables into a sequence of records (taken as input by the model): `data_new/*split*/trainData_*fold*.txt`
* Cleans summary tokens and substitutes any possible tokens with data variables(e.g., **2018** -> **templateValue[0][0]**): `data_new/*split*/trainSummary_*fold*.txt`
* Cleans the title tokens: `data_new/*split*/trainTitle_*fold*.txt`
* Labels the occurrences of records mentioned within the summary: `data_new/*split*/trainDataLabel_*fold*.txt`
* Labels the summary tokens which match a record: `data_new/*split*/trainSummaryLabel_*fold*.txt`
* Saves the gold summaries: `data_new/*split*/testOriginalSummary_*fold*.txt`

### Step2: Extract vocabulary for each split

```
cd etc
python extract_vocab.py --table ../data_new/valid/validData_1.txt --summary ../data_new/valid/validSummary_1.txt
python extract_vocab.py --table ../data_new/valid/validData_2.txt --summary ../data_new/valid/validSummary_2.txt
python extract_vocab.py --table ../data_new/valid/validData_3.txt --summary ../data_new/valid/validSummary_3.txt
python extract_vocab.py --table ../data_new/valid/validData_4.txt --summary ../data_new/valid/validSummary_4.txt
python extract_vocab.py --table ../data_new/valid/validData_5.txt --summary ../data_new/valid/validSummary_5.txt
python extract_vocab.py --table ../data_new/test/testData_1.txt --summary ../data_new/test/testSummary_1.txt
python extract_vocab.py --table ../data_new/test/testData_2.txt --summary ../data_new/test/testSummary_2.txt
python extract_vocab.py --table ../data_new/test/testData_3.txt --summary ../data_new/test/testSummary_3.txt
python extract_vocab.py --table ../data_new/test/testData_4.txt --summary ../data_new/test/testSummary_4.txt
python extract_vocab.py --table ../data_new/test/testData_5.txt --summary ../data_new/test/testSummary_5.txt
python extract_vocab.py --table ../data_new/train/trainData_1.txt --summary ../data_new/train/trainSummary_1.txt
python extract_vocab.py --table ../data_new/train/trainData_2.txt --summary ../data_new/train/trainSummary_2.txt
python extract_vocab.py --table ../data_new/train/trainData_3.txt --summary ../data_new/train/trainSummary_3.txt
python extract_vocab.py --table ../data_new/train/trainData_4.txt --summary ../data_new/train/trainSummary_4.txt
python extract_vocab.py --table ../data_new/train/trainData_5.txt --summary ../data_new/train/trainSummary_5.txt
```

It will generate vocabulary files for each of them:

* `data_new/*split*/trainData_*fold*.txt_vocab`
* `data_new/*split*/trainSummary_*fold*.txt_vocab`

### Step3: Binarize the data for each split

```
cd ../model
python preprocess_table_data.py --table ../data_new/valid/validData_1.txt --table_vocab ../data_new/valid/validData_1.txt_vocab --table_label ../data_new/valid/validDataLabel_1.txt
python preprocess_table_data.py --table ../data_new/valid/validData_2.txt --table_vocab ../data_new/valid/validData_2.txt_vocab --table_label ../data_new/valid/validDataLabel_2.txt
python preprocess_table_data.py --table ../data_new/valid/validData_3.txt --table_vocab ../data_new/valid/validData_3.txt_vocab --table_label ../data_new/valid/validDataLabel_3.txt
python preprocess_table_data.py --table ../data_new/valid/validData_4.txt --table_vocab ../data_new/valid/validData_4.txt_vocab --table_label ../data_new/valid/validDataLabel_4.txt
python preprocess_table_data.py --table ../data_new/valid/validData_5.txt --table_vocab ../data_new/valid/validData_5.txt_vocab --table_label ../data_new/valid/validDataLabel_5.txt
python preprocess_table_data.py --table ../data_new/test/testData_1.txt --table_vocab ../data_new/test/testData_1.txt_vocab --table_label ../data_new/test/testDataLabel_1.txt
python preprocess_table_data.py --table ../data_new/test/testData_2.txt --table_vocab ../data_new/test/testData_2.txt_vocab --table_label ../data_new/test/testDataLabel_2.txt
python preprocess_table_data.py --table ../data_new/test/testData_3.txt --table_vocab ../data_new/test/testData_3.txt_vocab --table_label ../data_new/test/testDataLabel_3.txt
python preprocess_table_data.py --table ../data_new/test/testData_4.txt --table_vocab ../data_new/test/testData_4.txt_vocab --table_label ../data_new/test/testDataLabel_4.txt
python preprocess_table_data.py --table ../data_new/test/testData_5.txt --table_vocab ../data_new/test/testData_5.txt_vocab --table_label ../data_new/test/testDataLabel_5.txt
python preprocess_table_data.py --table ../data_new/train/trainData_1.txt --table_vocab ../data_new/train/trainData_1.txt_vocab --table_label ../data_new/train/trainDataLabel_1.txt
python preprocess_table_data.py --table ../data_new/train/trainData_2.txt --table_vocab ../data_new/train/trainData_2.txt_vocab --table_label ../data_new/train/trainDataLabel_2.txt
python preprocess_table_data.py --table ../data_new/train/trainData_3.txt --table_vocab ../data_new/train/trainData_3.txt_vocab --table_label ../data_new/train/trainDataLabel_3.txt
python preprocess_table_data.py --table ../data_new/train/trainData_4.txt --table_vocab ../data_new/train/trainData_4.txt_vocab --table_label ../data_new/train/trainDataLabel_4.txt
python preprocess_table_data.py --table ../data_new/train/trainData_5.txt --table_vocab ../data_new/train/trainData_5.txt_vocab --table_label ../data_new/train/trainDataLabel_5.txt

python preprocess_summary_data.py --summary ../data_new/valid/validSummary_1.txt --summary_vocab ../data_new/valid/validSummary_1.txt_vocab --summary_label ../data_new/valid/validSummaryLabel_1.txt
python preprocess_summary_data.py --summary ../data_new/valid/validSummary_2.txt --summary_vocab ../data_new/valid/validSummary_2.txt_vocab --summary_label ../data_new/valid/validSummaryLabel_2.txt
python preprocess_summary_data.py --summary ../data_new/valid/validSummary_3.txt --summary_vocab ../data_new/valid/validSummary_3.txt_vocab --summary_label ../data_new/valid/validSummaryLabel_3.txt
python preprocess_summary_data.py --summary ../data_new/valid/validSummary_4.txt --summary_vocab ../data_new/valid/validSummary_4.txt_vocab --summary_label ../data_new/valid/validSummaryLabel_4.txt
python preprocess_summary_data.py --summary ../data_new/valid/validSummary_5.txt --summary_vocab ../data_new/valid/validSummary_5.txt_vocab --summary_label ../data_new/valid/validSummaryLabel_5.txt
python preprocess_summary_data.py --summary ../data_new/test/testSummary_1.txt --summary_vocab ../data_new/test/testSummary_1.txt_vocab --summary_label ../data_new/test/testSummaryLabel_1.txt
python preprocess_summary_data.py --summary ../data_new/test/testSummary_2.txt --summary_vocab ../data_new/test/testSummary_2.txt_vocab --summary_label ../data_new/test/testSummaryLabel_2.txt
python preprocess_summary_data.py --summary ../data_new/test/testSummary_3.txt --summary_vocab ../data_new/test/testSummary_3.txt_vocab --summary_label ../data_new/test/testSummaryLabel_3.txt
python preprocess_summary_data.py --summary ../data_new/test/testSummary_4.txt --summary_vocab ../data_new/test/testSummary_4.txt_vocab --summary_label ../data_new/test/testSummaryLabel_4.txt
python preprocess_summary_data.py --summary ../data_new/test/testSummary_5.txt --summary_vocab ../data_new/test/testSummary_5.txt_vocab --summary_label ../data_new/test/testSummaryLabel_5.txt
python preprocess_summary_data.py --summary ../data_new/train/trainSummary_1.txt --summary_vocab ../data_new/train/trainSummary_1.txt_vocab --summary_label ../data_new/train/trainSummaryLabel_1.txt
python preprocess_summary_data.py --summary ../data_new/train/trainSummary_2.txt --summary_vocab ../data_new/train/trainSummary_2.txt_vocab --summary_label ../data_new/train/trainSummaryLabel_2.txt
python preprocess_summary_data.py --summary ../data_new/train/trainSummary_3.txt --summary_vocab ../data_new/train/trainSummary_3.txt_vocab --summary_label ../data_new/train/trainSummaryLabel_3.txt
python preprocess_summary_data.py --summary ../data_new/train/trainSummary_4.txt --summary_vocab ../data_new/train/trainSummary_4.txt_vocab --summary_label ../data_new/train/trainSummaryLabel_4.txt
python preprocess_summary_data.py --summary ../data_new/train/trainSummary_5.txt --summary_vocab ../data_new/train/trainSummary_5.txt_vocab --summary_label ../data_new/train/trainSummaryLabel_5.txt

```
Outputs the training data:
* Data Records: `data_new/*split*/trainData_*fold*.txt.pth`
* Summaries: `data_new/*split*/trainSummary_*fold*.txt.pth`

Note: if you get a dictionary assertion error, then delete the old .pth files in data subfolders and try again
### Model Training
Replace the *fold* in the path for each fold:
```
MODELPATH=$PWD/model
export PYTHONPATH=$MODELPATH:$PYTHONPATH

python $MODELPATH/train.py

## main parameters
python model/train.py \
    --model_path "helen" \
    --exp_name "helen_exp" \
    --exp_id "helen1" \
    --train_cs_table_path data_new/train/trainData_1.txt.pth \
    --train_sm_table_path data_new/train/trainData_1.txt.pth \
    --train_sm_summary_path data_new/train/trainSummary_1.txt.pth \
    --valid_table_path data_new/valid/validData_1.txt.pth \
    --valid_summary_path data_new/valid/validSummary_1.txt.pth \
    --cs_step True \
    --lambda_cs "1" \
    --sm_step True \
    --lambda_sm "1" \
    --label_smoothing 0.05 \
    --sm_step_with_cc_loss False \
    --sm_step_with_cs_proba False \
    --share_inout_emb True \
    --share_srctgt_emb False \
    --emb_dim 512 \
    --enc_n_layers 1 \
    --dec_n_layers 6 \
    --dropout 0.1 \
    --save_periodic 40 \
    --batch_size 6 \
    --beam_size 4 \
    --epoch_size 1000 \
    --max_epoch 81 \
    --eval_meteor True \
    --sinusoidal_embeddings True \
    --encoder_positional_emb True \
    --gelu_activation True \
    --validation_metrics valid_mt_meteor
```

### Generation

Use the following commands to generate from the above models, 
replace the *fold* in the path for each fold:

```
python model/summarize.py 
  --model_path helen/helen_exp/helen1/periodic-80.pth 
  --table_path data_new/test/testData_1.txt \
  --output_path results_new/helen1/helen1.txt \
  --title_path data_new/test/testTitle_1.txt --beam_size 4 --batch_size 6
```

### Postprocessing after generation
Substitute any predicted data variables:

```
python etc/summaryComparison.py
```

### Evaluation

#### "Content Selection" evaluation
```
python studyOutcome/automatedEvaluation.py
```

#### Untrained automatic metrics evaluation (BLEU, ROUGE, METEOR)


```
python studyOutcome/evaluationMetrics.py
```

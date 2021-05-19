# HumanlikeHearing

A Python package for applying a range of psychometric tests on automatic speech recognition (ASR) systems. For more information on the psychometric tests and the ASR systems this toolbox supports, see our accompanying paper: 

The Psychometrics of Automatic Speech Recognition
Lotte Weerts, Stuart Rosen, Claudia Clopath, Dan F. M. Goodman
bioRxiv 2021.04.19.440438; doi: https://doi.org/10.1101/2021.04.19.440438


## Installation

The easiest way to install the toolbox is by installing the latest stable release that lives on PyPI:

```
pip install humanlikehearing
```

To ensure all dependencies are correctly installed, we recommend using Anaconda to install numpy and scipy beforehand.

To build the toolbox from source use:

```
python setup.py build
python setup.py install
```

If your installation went well, you should now be able to execute the demo script `run.py`:

```
python run.py \
  --asr_system_name TestASR 
  --dataset TestDataSet 
  --data_path . 
  --results_folder ../results 
  --sentences_per_condition 1
```

IMPORTANT: installing the toolbox DOES NOT install any of the automatic speech recognition systems - the sample script will run a dummy ASR system that always prints 'hello world'.

## Prepare ASR systems

By default, no ASR systems are included in the toolbox. However, the toolbox provides support for specific versions of three freely available ASR systems. If you just want to quickly test out the toolbox, we recommend installing Mozilla DeepSpeech v0.6.1, as it is the easiest to install. 

After installation, you can start running experiments by setting the --asr_system_name and --model_path accordingly:

```
python examples/run.py \
  --asr_system_name <ASR CLASS NAME> 
  --model_path <PATH TO ASR MODEL FILE>
  --dataset TestDataSet 
  --data_path . 
  --results_folder ../results 
  --sentences_per_condition 1
```

### MozillaDeepSpeech (LSTM model)

Installation instructions can be found on https://deepspeech.readthedocs.io/en/v0.6.1/USING.html

This code assumes the model follows Mozilla DeepSpeech version 6.1 and may not work for later models! When defining `model_path` refer to the unzipped directory (e.g. `/path/to/downloads/deepspeech-0.6.1-models`). 

### Vosk's Kaldi nnet3 model (DNN-HMM model)

Installation instructions can be found on https://alphacephei.com/vosk/install

The Vosk model used in the paper is vosk-model-en-us-daanzu-20200905 and can be downloaded here: https://alphacephei.com/vosk/models

Note that to be able to run this model, you also need to  install Kaldi: http://www.kaldi-asr.org/doc/install.html

When defining `model_path` refer to the unzipped directory (e.g. `/path/to/downloads/vosk-model-en-us-daanzu-20200905`). 

### Fairseq's Wav2vec 2.0 (CNN-Transformer model)

Installation instructions can be found on: https://github.com/pytorch/fairseq/tree/828960f5dace4787ad81aeadca60043c907adc67/examples/wav2vec

The Wav2Vec model used in the paper is the Wav2Vec 2.0 Large model trained for 960 hours. 

When defining `model_path` refer to the `.pt` file of the model (e.g. `/path/to/downloads/wav2vec_big_960h.pt`). Note that it is assumed that in the same folder, a `dict.ltr.txt` file is present. This file can be downloaded here: https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt

## Prepare DataSets

The toolbox supports the use of two freely available speech datasets, the ARU speech corpus (which contains recordings of the IEEE sentences) and the LibriSpeech dataset (which contains recordings of audiobooks). We generally recommend the ARU speech corpus for testing as it is most similar to the type of data humans tend to be tested on, and not all experiments are currently compatible with the LibriSpeech dataset (but will be in the future). 

### ARU Speech Corpus
The ARU dataset can be downloaded here: http://datacat.liverpool.ac.uk/681/

To run an experiment on the ARU speech corpus: 

```
python examples/run.py \
  --dataset ARUDataSet 
  --data_path /your/path/to/ARU_Speech_Corpus_v1_0
  --results_folder ../results 
  --sentences_per_condition 100
```

### LibriSpeech Corpus
The LibriSpeech test data can be downloaded here: https://www.openslr.org/12

Note: We recommend to only use the "test-clean.tar.gz" subset of the LibriSpeech data set, as many freely available ASR systems are trained using LibriSpeech, so testing using the training data will overestimate the ASR performance.

To run an experiment on the LibriSpeech corpus:

```
python examples/run.py \
  --dataset LibriSpeechDataSet 
  --data_path /your/path/to/test-clean
  --results_folder ../results 
  --sentences_per_condition 100
```

## Run an experiment

To run an experiment, you can either use run.py to load the correct asr system and data set and write the outputs to a results folder. By default, run.py will run all experiments described in the paper:

```
python examples/run.py \
  --asr_system_name <ASR SYSTEM CLASS> 
  --model_path <PATH TO ASR MODEL>
  --dataset <DATA SET CLASS NAME>  
  --data_path <PATH TO DATA>
  --results_folder <RESULTS FOLDER>
  --sentences_per_condition 100
```

Here, --asr_system_name, --model_path, --dataset and --data_path can be defined as described above. The --results_folder indicates the folder in which experimental outcomes will be stored as pandas tables. --sentences_per_condition indicates how many sentences are used per condition. For most experiments, in particular the SRT experiments, you want this number to be at least 20, but closer to 100 will give you a better view on the model performance. 

If you only want to run a subset of the experiments or if you want to change any parameters, you can simply edit run.py as desired. 

## Analyse your experimental results

To view the outcomes of your experiments, locate your experiment folder in your results folder, which are organised as `results/test_report_<ASRNAME>_<TIMESTAMP>/<EXPERIMENT CLASS>_<RESULTS TYPE>_<TIMESTAMP>`. Here, `<RESULTS_TYPE>` is usually 'standard', but in some cases may indicate a sub-experiment (e.g. a clipping experiment will have a 'peak' and 'center' results type). 

To load your experimental outcomes, you can use pandas:

```
import pandas as pd
results = pd.read_pickle('path/to/experiment/results.pk1')
```

In most cases, it will be relatively straight forward to analyse the outcomes. However, in the case of speech reception threshold (SRT) experiments, one extra step of analysis is required to obtain the SRTs from the results file. See `examples/srt_analysis.ipynb` for an example of how to obtain SRT results. 

# Citing

If you wish to cite HumanlikeHearing in a scholarly work, please cite the following:

The Psychometrics of Automatic Speech Recognition
Lotte Weerts, Stuart Rosen, Claudia Clopath, Dan F. M. Goodman
bioRxiv 2021.04.19.440438; doi: https://doi.org/10.1101/2021.04.19.440438
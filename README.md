# RNN-Transducer

This is a PyTorch implementation of [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/pdf/1211.3711.pdf) speech recognition paper 

```
@article{DBLP:journals/corr/abs-1211-3711,
  author    = {Alex Graves},
  title     = {Sequence Transduction with Recurrent Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1211.3711},
  year      = {2012},
  url       = {http://arxiv.org/abs/1211.3711},
  eprinttype = {arXiv},
  eprint    = {1211.3711},
  timestamp = {Mon, 13 Aug 2018 16:48:55 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1211-3711.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
# Train on your data
In order to train the model on your data follow the steps below 
### 1. data preprocessing 
* prepare your data and make sure the data is formatted in an CSV format as below 
```
audio_path,text,duration
file/to/file.wav,the text in that file,3.2 
```
* make sure the audios are MONO if not make the proper conversion to meet this condition

### 2. Setup development environment
* create environment 
```bash
python -m venv env
```
* activate the environment
```bash
source env/bin/activate
```
* install the required dependencies
```bash
pip install -r requirements.txt
```

### 3. Training 
* update the config file if needed
* train the model 
  * from scratch 
  ```bash
  python train.py
  ```
  * from checkpoint 
  ```
  python train.py checkpoint=path/to/checkpoint tokenizer.tokenizer_file=path/to/tokenizer.json
  ```

# TODO
- [ ] adding the inference module 
- [ ] Adding Demo

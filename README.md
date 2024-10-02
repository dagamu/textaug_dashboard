# Text Augmentation Dashboard
## Data Augmentation with LLMs for MLC Tasks

- Requirements
    - streamlit==1.38.0
    - pandas
    - scikit-learn
    - scikit-multilearn
    - matplotlib
    - nltk
    - neattext
    - nlpaug (*)
        
---

## Quick Start

```console
python
> import nltk
> nltk.download('stopwords')
> nltk.download('averaged_perceptron_tagger_eng')
```
```console
pip install -r requirements.txt
pip install numpy requests nlpaug
```

--- 

run with `streamlit run dashboard.py` or `python -m streamlit run dashboard.py`
# Text Augmentation Dashboard
## Data Augmentation with LLMs for MLC Tasks

- Requirements
    - streamlit
    - streamlit-option-menu
    - pandas
    - scikit-learn
    - scikit-multilearn
    - matplotlib
    - nltk
    - neattext
    - nlpaug (*)
        
---

```bash
python
> import nltk
> nltk.download('stopwords')
> nltk.download('averaged_perceptron_tagger_eng')
```
```bash
pip install -r requirements.txt
pip install numpy requests nlpaug
```

--- 

run with `streamlit run dashboard_home.py` or `python -m streamlit run dashboard_home.py`
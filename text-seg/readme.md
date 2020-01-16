# Semantic Text Spliter
To get the keywords of a paragraph for searching proper images.

## Dependencies
Use the default pytorch conda ve provided: conda activate pytorch
```
pip install --user transformers nltk pytorch-pretrained-bert

git clone https://github.com/NVIDIA/apex
cd apex
pip install --user -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
In python, run this
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

##            Example Usage           
### Assumed input text  
```
text = """The name machine learning was coined in 1959 by Arthur Samuel. Tom M. Mitchell provided a widely quoted, more formal definition of the algorithms studied in the machine learning field: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E." """
```

### Configuration
text_seg.py must be in the same folder as the code who calls it
```
from text_seg import TextSplitter
model_dir = "<your_path>/models/"
token_model_path = "<your_path>/model.pt"
```

### Initialize one splitter
It takes time, so reuse it!
```
splitter = TextSplitter(model_dir, token_model_path) 
```

### Reuse the same splitter multiple times
```
segments, keywords = splitter.split(text)
```
## Example outputs:

### segments
```
[
	'The name machine learning was coined in 1959 by Arthur Samuel.', 
	'Tom M. Mitchell provided a widely quoted, more formal definition of the algorithms studied in the machine learning field: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E."'
]

```

### keywords
```
[['machine', 'learning'], ['experience', 'machine', 'learning']]
```
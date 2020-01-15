# Text Splitter
## Dependencies
* default pytorch conda ve provided: conda activate pytorch
* pip install --user transformers nltk
* In python, run this
```
import nltk
nltk.download('punkt')
```


##            Example Usage             

text_seg.py must be in the same folder as the code who calls it
```
text = r"This is the text to be split. Many sentences. Sometimes '\n'."

from text_seg import TextSplitter
model_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/pregen/models/"

```

### Initialize one splitter
It takes time, so reuse it!
```
splitter = TextSplitter(model_dir) 
```


### Reuse the same splitter multiple times
```
segments, key_phrases = splitter.split(text)
```
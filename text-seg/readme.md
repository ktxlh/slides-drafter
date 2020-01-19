# Text to Slides Contents
1. To split the paragraph semantically into ``segments``.
2. To extract ``keywords`` of each semantical segment for searching proper images. We trained our model with the code from [this repo](https://github.com/ibatra/BERT-Keyword-Extractor).
3. To get the key sentences of text as a candidate of the title of each slide (``subtitles``) and the whole slide deck (``title``). We adopted TextRank as a tool from [this library](https://github.com/miso-belica/sumy).

## Dependencies
* Used the default pytorch conda provided: ``conda activate pytorch``
* The following should be installed
```
pip install --user transformers nltk pytorch-pretrained-bert summa
```
* The following seems to work for linux only. See [the original repo](https://github.com/NVIDIA/apex). 
```
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
text = """Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet's leading subsidiary and will continue to be the umbrella company for Alphabet's Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet."""
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
title, segments, keywords, subtitles = splitter.split(text)
```
* text: str -- Normal text input
* title: str -- The key sentence of the whole text or "Title"
* segments: list(str) -- Each str is a semantic segment.
* keywords: list(list(str)) -- Each list corresponds to a segment, containing 0 or more keywords
* subtitles: list(str) -- Each str may be "Title" or one key sentence.

## Example outputs:

### segments
```
[
	'Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock.', 
	'They incorporated Google as a California privately held company on September 4, 1998, in California.', 
	'Google was then reincorporated in Delaware on October 22, 2002.', 
	'An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex.', 
	"In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet's leading subsidiary and will continue to be the umbrella company for Alphabet's Internet interests.", 
	'Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet.'
]
```

### keywords
```
[[], ['Google'], [], ['initial', 'offering', 'public'], [], ['CEO']]
```

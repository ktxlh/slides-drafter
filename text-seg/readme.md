# Text Splitter
## Dependencies
* default pytorch conda ve provided: conda activate pytorch
* pip install --user transformers nltk pytorch-pretrained-bert
* In python, run this
```
import nltk
nltk.download('punkt')
```


##            Example Usage           
### Assumed input text  
```
text = """Some students space paragraphs, trying to separate points when the process of writing is over. This is a major mistake. How much easier your writing would become if you looked at it from another angle! It is reasonable to use different types of paragraphs WHILE you are writing. In case you follow all the rules, you'll have no difficulty in bringing your message across to your reader.
If you browse for ‘the types of paragraphs' you'll be surprised how many results you'll get. Among others, the four following types should be distinguished: descriptive, expository, narrative, and persuasive paragraphs. Mastering these types will help you a lot in writing almost every type of texts.
Descriptive: These paragraphs have four main aims. First of all, they naturally describe something or somebody, that is conveying the information. Secondly, such paragraphs create powerful images in the reader's mind. Thirdly, they appeal to the primary senses of vision, hearing, touch, taste, and smell, to get the maximum emotional response from the reader. And finally, they increase the dynamics of the text. Some grammar rules may be skipped in descriptive paragraphs, but only for the sake of imagery.
Expository: It is not an easy task to write an expository paragraph, especially if you are an amateur in the subject. These paragraphs explain how something works or what the reader is to do to make it work. Such paragraphs demand a certain knowledge. Nevertheless, writing them is a great exercise to understand the material, because you keep learning when you teach."""
```

text_seg.py must be in the same folder as the code who calls it
```
from text_seg import TextSplitter
model_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/pregen/models/"
token_model_path = "~/BERT-Keyword-Extractor/model.pt"

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
['Some students space paragraphs, trying to separate points when the process of writing is over.', 'This is a major mistake. How much easier your writing would become if you looked at 
it from another angle! It is reasonable to use different types of paragraphs WHILE you are writing.', "In case you follow all the rules, you'll have no difficulty in bringing your mess
age across to your reader.", "If you browse for ‘the types of paragraphs' you'll be surprised how many results you'll get. Among others, the four following types should be distinguishe
d: descriptive, expository, narrative, and persuasive paragraphs.", 'Mastering these types will help you a lot in writing almost every type of texts.', 'Descriptive: These paragraphs h
ave four main aims. First of all, they naturally describe something or somebody, that is conveying the information.', "Secondly, such paragraphs create powerful images in the reader's 
mind.", 'Thirdly, they appeal to the primary senses of vision, hearing, touch, taste, and smell, to get the maximum emotional response from the reader.', 'And finally, they increase th
e dynamics of the text. Some grammar rules may be skipped in descriptive paragraphs, but only for the sake of imagery.', 'Expository: It is not an easy task to write an expository para
graph, especially if you are an amateur in the subject.', 'These paragraphs explain how something works or what the reader is to do to make it work.', 'Such paragraphs demand a certain
 knowledge.', 'Nevertheless, writing them is a great exercise to understand the material, because you keep learning when you teach.']

```

### keywords
```
[['writing'], [], ['writing'], [], ['message'], ['browse'], ['descriptive'], ['writing'], [], ['response', 'emotional', 'vision', 'dynamics'], ['grammar'], ['amateur'], ['certain'], ['
material', 'learning']]
```
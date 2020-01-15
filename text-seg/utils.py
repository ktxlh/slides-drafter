import os
import json
from nltk.tokenize import sent_tokenize

def traverse_json_dir(json_dir, toke_to_sent, limit_paragraphs):

    def remove_non_printable(s):
        s = s.encode('ascii', errors='ignore').decode('ascii')
        s = '\n'.join([ss for ss in s.split('\n') if len(ss) > 40])
        return s

    sections = []
    for root, dirs, files in os.walk(json_dir):
        print("# of json files in total:",len(files))
        files.sort()
        for fname in files:
            obj = json.load(open(os.path.join(json_dir, fname)))
            for secs in obj['now']['sections']:
                text = remove_non_printable(secs['text'])
                if len (text) > 40:
                    if toke_to_sent:
                        sentences = sent_tokenize(text)
                        sections.append(sentences)
                    else:
                        sections.append(text)
            if limit_paragraphs > 0 and len(sections) >= limit_paragraphs: ### TODO Use more data later
                break

    print("# of sections loaded:", len(sections))
    return sections
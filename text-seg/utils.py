from nltk.tokenize import sent_tokenize

def traverse_json_dir(json_dir, toke_to_sent, limit_paragraphs):

    def remove_non_printable(s):
        return s.encode('ascii', errors='ignore').decode('ascii')

    sections = []
    for root, dirs, files in os.walk(json_dir):
        print("# of json files in total:",len(files))
        files.sort()
        for fname in files:
            obj = json.load(open(os.path.join(json_dir, fname)))
            for secs in obj['now']['sections']:
                text = remove_non_printable(secs['text'])
                if len (text) > 0:
                    sentences = sent_tokenize(text)
                    if len(sentences) > 10:
                        continue # Some tables are weird <1>
                    if toke_to_sent:
                        sections.append(sentences)
                    else:
                        sections.append(text)
            if limit_paragraphs > 0 and len(sections) >= limit_paragraphs: ### TODO Use more data later
                break

    print("# of sections loaded:", len(sections))
    return sections
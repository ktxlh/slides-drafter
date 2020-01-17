import nltk
import argparse
from text_seg import TextSplitter



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file", default='./input.txt', help='input_file')
    parser.add_argument('-output_file', default='./seg.txt', help='seg_file')
    parser.add_argument('-base_dir', default='../', help='base_dir')
    args = parser.parse_args()

    input_file = args.input_file
    seg_file = args.output_file
    base_dir = args.base_dir

    with open(input_file) as f:
        text = f.read()
        f.close()


    model_dir = base_dir + "models/"
    token_model_path = base_dir + "model.pt"
    splitter = TextSplitter(model_dir, token_model_path) 
    title, segments, keywords, subtitles = splitter.split(text)

    # segments = ['1111111111', '2222222222', '33333333333333', '222222222222']
    # keywords = [['ww', 'eee'], [''], ["rr"], ["3"]]

    # print(segments)
    # print(keywords)


    with open(seg_file, 'w') as f:
        print(title, file=f)
        for seg, kw, st in zip(segments, keywords, subtitles):
            if st == '' and len(kw) == 0:
                continue
            if st == '':
                print(" @", file=f, end='')
            else:
                print(st + '@', file=f, end='')
            if len(kw) == 0:
                print(seg, file=f)
            else:
                print('@'.join(kw), file=f)
        
        f.close()
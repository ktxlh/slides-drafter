import nltk
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file", default='test.txt', help='input_file')
    parser.add_argument('-output_file', default='output.txt', help='output_file')
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    with open(input_file) as f:
        text = f.read()
        f.close()

    # from text_seg import TextSplitter
    # model_dir = "/home/shanglinghsu/ml-camp/wiki-vandalism/mini-json-raw/pregen/models/"
    # splitter = TextSplitter(model_dir) 
    # segments = splitter.split(text)

    segments = ['1111', '2222', '3333', '4444444']

    with open(output_file, 'w') as f:
        for sentence in segments:
            print(sentence, file=f)
        
        f.close()


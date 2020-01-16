import nltk
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seg_file", default='test.txt', help='seg_file')
    parser.add_argument("-title_file", default='test.txt', help='title_file')
    parser.add_argument('-output_file', default='output.txt', help='output_file')
    args = parser.parse_args()

    output_file = args.output_file
    segments = ['1111', '2222', '3333', '4444444']

    with open(output_file, 'w') as f:
        print("Final Slides", file=f)
        for sentence in segments:
            print(sentence, file=f)
        
        f.close()


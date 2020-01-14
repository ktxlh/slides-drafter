from jieba.analyse import *
data = open('usercontent.txt').read()#读取文件
for keyword,weight in extract_tags(data,withWeight = True):
    print('%s %s' %(keyword,weight))

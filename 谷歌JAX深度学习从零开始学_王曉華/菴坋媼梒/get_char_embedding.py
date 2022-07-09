import re
import csv
#from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

#对英文文本进行数据清洗
#stoplist = stopwords.words('english')
def text_clear(text):
    text = text.lower()  				#将文本转化成小写字母
    text = re.sub(r"[^a-z]"," ",text)   	#替换非标准字符，^是求反操作
    text = re.sub(r" +", " ", text) 	#替换多重空格
    text = text.strip()  				#取出首尾空格
    text = text.split(" ")
    #text = [word for word in text if word not in stoplist]    	#去除停用词
    text = [PorterStemmer().stem(word) for word in text]    	#还原词干部分
    text.append("eos")                 				#添加结束符
    text = ["bos"] + text             				#添加开始符
    return text
#对标题进行处理
def text_clearTitle(text):
    text = text.lower()  							#将文本转化成小写字母
    text = re.sub(r"[^a-z]"," ",text)   				#替换非标准字符，^是求反操作
    text = re.sub(r" +", " ", text) 				#替换多重空格
    #text = re.sub(" ", "", text) 					#替换隔断空格
    text = text.strip()  							#取出首尾空格
    text = text + " eos"                 				#添加结束符
    return text
#生成标题的独热编码标签
def get_label_one_hot(list):
    values = np.array(list)
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]
#生成文本的独热编码矩阵
def get_one_hot(list,alphabet_title = None):
    if alphabet_title == None: 				#设置字符集
        alphabet_title = "abcdefghijklmnopqrstuvwxyz "
    else:alphabet_title = alphabet_title
    values = np.array(list) 					#获取字符数列
    n_values = len(alphabet_title) + 1  	#获取字符表长度
    return np.eye(n_values)[values]
#获取文本在词典中的位置列表
def get_char_list(string,alphabet_title = None):
    if alphabet_title == None:
        alphabet_title = "abcdefghijklmnopqrstuvwxyz "
    else:alphabet_title = alphabet_title
    char_list = []
    for char in string:   					#获取字符串中的字符
        num = alphabet_title.index(char) 	#获取对应位置
        char_list.append(num) 				#组合位置编码
    return char_list
#生成文本矩阵
def get_string_matrix(string):
    char_list = get_char_list(string)
    string_matrix = get_one_hot(char_list)
    return string_matrix
#获取补全后的文本矩阵
def get_handle_string_matrix(string,n = 64):
    string_length= len(string)
    if string_length > 64:
        string = string[:64]
        string_matrix = get_string_matrix(string)
        return string_matrix
    else:
        string_matrix = get_string_matrix(string)
        handle_length = n - string_length
        pad_matrix = np.zeros([handle_length,28])
        string_matrix = np.concatenate([string_matrix,pad_matrix],axis=0)
        return string_matrix
#获取数据集
def get_dataset(file_path = "./train.csv"):
    agnews_label = []
    agnews_title = []
    agnews_train = csv.reader(open(file_path,"r"))
    for line in agnews_train:
        agnews_label.append(np.int(line[0]))
        agnews_title.append(text_clearTitle(line[1]))
    train_dataset = []
    for title in agnews_title:
        string_matrix = get_handle_string_matrix(title)
        train_dataset.append(string_matrix)
    train_dataset = np.array(train_dataset)
    label_dataset = get_label_one_hot(agnews_label)
    train_dataset = np.expand_dims(train_dataset,axis=-1)

    return train_dataset,label_dataset


if __name__ == '__main__':
    train_dataset, label_dataset = get_dataset()

    print(train_dataset.shape)
    print(label_dataset.shape)



"""
    1. 应用纹理特征；
    2. 分词，扩充语料的范围。
    2. 修补字频分布不均匀的问题。

"""
#%%
import os
import jieba
from functools import reduce
import operator
import pickle
from collections import Counter
import pandas as pd
import math
class Sematics:

    def __init__(self, corps_file_list=None, out_list_corps="list_corps.txt"):
        """
            加载所需要的各种原料。out_list_corps是相对路径。
        """
        # 加载纹理属性。        
        self.dfTexture = pd.read_pickle("data/base_texture.pkl")

        self.out_list_corps = out_list_corps
        self.__genCorps__(corps_file_list)
        self.low_f = []
        self.high_f = []
        tmp = {'0': 1452990, '1': 702625, '2': 523600, '3': 352275, '7': 281820, '税': 262955, '5': 209055, '6': 202125, '9': 197505, '-': 177870, '8': 170940, '4': 162855, '(': 148995, '.': 143605, ')': 125125, '人': 122045, ',': 117040, '号': 115500, '退': 115500, '纳': 114730, '名': 114730, '称': 114730, '金': 113960, '额': 113960, '品': 88935, '房': 80850, '证': 75845, '种': 57750, '原': 57750, '凭': 57750, '元': 57750, '￥': 57750, '识': 57750, '别': 57750, '目': 57750, '入': 57750, '库': 57750, '日': 57750, '期': 57750, '实': 57750, '缴': 57750, '合': 56210, '计': 56210, '万': 52745, '仟': 48125, '契': 46200, '买': 45430, '住': 44660, '伍': 43890, '整': 43890, '佰': 40425, '壹': 31570, '商': 31570, '存': 29645, '量': 29645, '柒': 26180, '贰': 24640, '花': 24255, '拾': 24255, '叁': 24255, '权': 24255, '印': 23870, '卖': 23870, '零': 21945, '、': 21945, '肆': 21560, '利': 18480, '许': 18095, '可': 18095, '照': 18095, '捌': 16940, '玖': 15015, '所': 13860, '得': 13860, '附': 13475, '角': 12705, '转': 12705, '分': 11935, '育': 11550, '陆': 11550, '教': 10780, '加': 10780, '增': 10780, '值': 10780, '王': 7315, '屋': 7315, '个': 6930, '让': 6930, '张': 5775, '产': 5775, '移': 5775, '书': 5775, '据': 5775, '地': 5390, '方': 5390, '费': 5390, '建': 5005, '城': 4620, '%': 3850, '市': 3465, '维': 3080, '李': 3080, 'X': 3080, '护': 2695, '设': 2695, '赵': 2310, '非': 2310, '杨': 1925, '刘': 1925, '马': 1925, '高': 1925, '县': 1925, '镇': 1925, '筑': 1925, '物': 1925, '周': 1540, '徐': 1540, '成': 1540, '华': 1540, '朱': 1540, '晓': 1540, '孙': 1540, '征': 1540, '其': 1540, '他': 1540, '文': 1540, '尹': 1540, '二': 1540, '丽': 1155, '新': 1155, '海': 1155, '耀': 1155, '志': 1155, '远': 1155, '林': 1155, '陈': 1155, '梅': 1155, '晶': 1155, '飞': 1155, '光': 1155, '办': 1155, '龙': 1155, '杰': 1155, '斌': 1155, '永': 1155, '英': 1155, '伟': 1155, '强': 1155, '颖': 770, '景': 770, '涛': 770, '刚': 770, '曹': 770, '劼': 770, '来': 770, '超': 770, '佳': 770, '立': 770, '彤': 770, '娜': 770, '雪': 770, '静': 770, '史': 770, '德': 770, '青': 770, '韩': 770, '胡': 770, '楠': 770, '洁': 770, '廖': 770, '姜': 770, '郭': 770, '贾': 770, '国': 770, '夏': 770, 'J': 770, 'C': 770, 'D': 770, 'H': 770, '区': 770, '公': 770, '宁': 770, '思': 770, '俊': 770, '瑞': 770, '勇': 770, '淑': 770, '鹏': 770, '玮': 770, '明': 770, '月': 770, '芳': 770, '孟': 770, '栋': 770, '凤': 770, '萍': 770, '民': 385, '币': 385, '石': 385, '峰': 385, '鑫': 385, '修': 385, '蔚': 385, '枫': 385, '罗': 385, '霍': 385, '克': 385, '渊': 385, '猛': 385, '敏': 385, '训': 385, '卫': 385, '兵': 385, '洪': 385, '锋': 385, '显': 385, '蕊': 385, '齐': 385, '波': 385, '庹': 385, '虎': 385, '娅': 385, '妤': 385, '茜': 385, '培': 385, '涵': 385, '姚': 385, '世': 385, '缘': 385, '鸣': 385, '燕': 385, '奕': 385, '达': 385, '邹': 385, '祥': 385, '臣': 385, '韦': 385, '颢': 385, '福': 385, '荣': 385, '玥': 385, '宋': 385, '伯': 385, '翁': 385, '苗': 385, '顺': 385, '双': 385, '云': 385, '影': 385, '宜': 385, '蒋': 385, '琴': 385, '温': 385, '婧': 385, '索': 385, '进': 385, '钦': 385, '宝': 385, '珍': 385, '南': 385, '连': 385, '常': 385, '洞': 385, '赜': 385, '锡': 385, '潘': 385, '洋': 385, '黄': 385, '占': 385, '霞': 385, '睿': 385, '娄': 385, '正': 385, '嘉': 385, '宗': 385, '辉': 385, '钰': 385, '芬': 385, '田': 385, '玉': 385, '山': 385, '董': 385, '庞': 385, '魏': 385, '娟': 385, '航': 385, '美': 385, '莹': 385, '司': 385, '卢': 385, '翟': 385, '磊': 385, '运': 385, '琼': 385, '曙': 385, '坚': 385, '喜': 385, '柴': 385, '启': 385, '欧': 385, '桐': 385, '心': 385, '庆': 385, '凯': 385, '春': 385, '玲': 385, '牛': 385, '一': 385, '村': 385, '苏': 385, '博': 385, '晖': 385, '悦': 385, '冯': 385, '乐': 385, '天': 385, '然': 385, '任': 385, '力': 385, '群': 385, '范': 385, '浩': 385, '淼': 385, '清': 385, '哲': 385, '江': 385, '乔': 385, '孔': 385, '泽': 385, '邓': 385, '瀚': 385, '安': 385, '跃': 385, '梁': 385, '珊': 385, '解': 385, '旭': 385, '晨': 385, '祁': 385, '中': 385, '阳': 385, '谢': 385, '桂': 385, '彭': 385, '卓': 385, '秦': 385, '泓': 385, '亚': 385, '坤': 385, '芝': 385, '宾': 385, '肖': 385, '亮': 385, '郄': 385, '手': 385, '保': 385, '障': 385, '性': 385}
        self.counter = Counter(tmp)
        # 加载字频分布
        if os.path.exists("word_distribution.pkl"):
            pkl_wd = open("word_distribution.pkl", "rb")
            self.counter = pickle.load(pkl_wd)

    def haveImageLowChaFreqFix(self, parameter_list):
        """
            查找字频缺失，输出是，一种补齐的语料。如何平衡字频问题。。。
            现在情况分化为两种情况
            1. （有图片）低频增补，
            2. （无图片），纯textTure控制生产。。。
            总体，进行隔离进行。
        """



    def __genCorps__(self, corps_file_list):
        """
            扩充原有list的语料。。原有的东西，可以是file，list等。。这里确定为是原有的file_list...
            file中的corps以每条一行的的状态存在。。。

        """
        self.origin_file_line = []
        self.corp_list = []
        if corps_file_list is not None:
            for file in corps_file_list:
                # file 是一个文件。
                # 不存储衍生特征。。。
                file_name = os.path.basename(file)
                with open(file, encoding="utf-8") as f:
                    t_line_list = [str(e).strip() for e in f.readlines()]
                    t_line_fenci = [list(jieba.cut(e)) for e in t_line_list]
                self.origin_file_line.append([file_name, t_line_list, t_line_fenci])
                self.corp_list.extend(t_line_list)
                self.corp_list.extend(reduce(operator.concat, t_line_fenci))
        # 获取到了所有的语料的情况。。
        with open(self.out_list_corps, "w", encoding="utf-8") as f:
            for line in self.corp_list:
                f.write(str(line).strip() + "\n")

    def departHFLF(self, train_file=None, label_file_list=None):
        """
            首先定义，高低频率。高频无法确定。但是，小于某个阈值，就一定是有问题的。按照字典大小，和图片总数量来说，是一个很小的值。
            路径保持一致。
        """
        if train_file is None:
            print("训练数据的标签不能为空")
            exit(1)
        with open(train_file, encoding='utf-8') as f:
            self.all_image_num = len(f.readlines())
        # 在这里进行统计分母的事情。
        self.HFLF_threshold = math.floor(self.all_image_num * 1.0 / len(self.counter))
        # 维护一个低频list。。
        for key, value in self.counter.items():
            if value < self.HFLF_threshold:
                self.low_f.append(key)
            else:
                self.high_f.append(key)        
        # 含有低频字的，就算是低频数据集。。。
        for file in label_file_list:
            with open(file, encoding="utf-8") as f:
                tmp_line_l = [str(e).strip() for e in f.readlines()]
            with open(file + "_filter_l", "w", encoding="utf-8") as lf, open(file + "_filter_h", "w", encoding="utf-8") as hf:
                for line in tmp_line_l:
                    fname = line.split(" ")[0]
                    content = line[len(fname):].strip()
                    # 判断content是否含有低频字，或者，低频字是否in content中的。
                    if any(ch in content for ch in self.low_f):
                        lf.write(line + "\n")
                    else:
                        hf.write(line + "\n")
                # 针对每一个文件file。。
 
x = Sematics(corps_file_list=["/Users/GuoLiuFang/Downloads/label_tmp_all20190311.txt_filter_l.txt"])
# x.departHFLF(train_file="/Users/GuoLiuFang/Downloads/tmp_labels.txt",label_file_list=["/Users/GuoLiuFang/Downloads/tmp_labels.txt", "/Users/GuoLiuFang/Downloads/label_tmp_all20190311.txt"])

#%%
print("高频字的数量", len(x.high_f))
print("低频字的数量", len(x.low_f))
print(x.HFLF_threshold)

#%%
x.dfTexture.head()
print(len(x.dfTexture))
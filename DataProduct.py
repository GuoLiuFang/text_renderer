import os
import subprocess
from PIL import Image
import shutil
from collections import Counter
import glob
import pandas as pd
import numpy as np
import math
import statistics
import pickle
import time
from multiprocessing import Pool
class gexinghuaRunner:
    """
        job_name中不能含有点.号。
    """

    def __base_have_image__(self, image_dir_path, train_file, corpus_dir, per_img_num, conf, tmp_prefix, job_name, is_fix):

        with open(os.path.join(image_dir_path, train_file), encoding='utf-8') as f:
            test_image_list = [str(line).strip() for line in f.readlines()]
        with open(f"{job_name}-bad_label_case.txt", "w", encoding="utf-8") as sf:
            for line in test_image_list:
                fname = line.split(" ")[0]
                content = line[len(fname):]
                content = content.strip()

                tmpimg = Image.open(os.path.join(image_dir_path, fname))
                tmp_w = tmpimg.size[0]
                tmp_h = tmpimg.size[1]

                # 如果字符长度的2倍，大于，宽度除以高度的话，才可以进行数据生产：497，25。字符长度为10的时候，2*10>(497/25)
                if (len(content) * 2) > (tmp_w * 1.0 / tmp_h):
                    corpus_f = os.path.join(corpus_dir, fname)
                    if os.path.exists(corpus_f):
                        shutil.rmtree(corpus_f)
                    os.makedirs(corpus_f)
                    with open(f"{corpus_f}/{fname}.txt", "w", encoding='utf-8') as tmpf:
                        tmpf.write(f"{content}\n")

                    tmpdict = dict(strict="", 
                                tag=f"{job_name}-{fname}.base",
                                num_img=f"{per_img_num[0]}",
                                config_file=f"{conf}",
                                corpus_dir=f"{tmp_prefix}{corpus_f}",
                                fonts_list="data/fonts_list/chn.txt",
                                corpus_mode="list",
                                output_dir=f"{tmp_prefix}{self.o_dir}")
                    tmpdict['img_width'] = tmp_w
                    tmpdict['img_height'] = tmp_h
                    self.configs.append((tmpdict, True))

                    if not is_fix:
                        tmpimg.crop([0, 0, tmp_w * 0.05 + 1, tmp_h]).convert('RGB').save(f"data/bg_base/{job_name}-{fname}-1.jpg")
                        tmpimg.crop([tmp_w * 0.95 - 1, 0, tmp_w, tmp_h]).convert('RGB').save(f"data/bg_base/{job_name}-{fname}-3.jpg")

                        self.widthList.append(tmp_w)
                        self.heightList.append(tmp_h)
                        self.labelLenList.append(len(content))
                        self.textureList.append([tmp_w, tmp_h, content, f"data/bg_base/{job_name}-{fname}"])

                    x1 = dict(strict="",
                                tag=f"{job_name}-{fname}.line",
                                num_img=f"{per_img_num[1]}",
                                config_file=f"{conf}",
                                corpus_dir=f"{tmp_prefix}{corpus_f}",
                                fonts_list="data/fonts_list/chn.txt",
                                corpus_mode="list",
                                output_dir=f"{tmp_prefix}{self.o_dir}")
                    x1['config_file'] = 'configs/mix_data_line.yaml'
                    x1['img_width'] = tmp_w
                    x1['img_height'] = tmp_h
                    self.configs.append((x1, True))

                    x2 = dict(strict="",
                                tag=f"{job_name}-{fname}.blur",
                                num_img=f"{per_img_num[2]}",
                                config_file=f"{conf}",
                                corpus_dir=f"{tmp_prefix}{corpus_f}",
                                fonts_list="data/fonts_list/chn.txt",
                                corpus_mode="list",
                                output_dir=f"{tmp_prefix}{self.o_dir}")
                    x2['config_file'] = 'configs/mix_data_blur.yaml'
                    x2['img_width'] = tmp_w
                    x2['img_height'] = tmp_h
                    self.configs.append((x2, True))

                    if "   " in content:
                        sf.write(f"configs/mix_data_space.yaml-content three space==={line}\n")
                    else:
                        x3 = dict(strict="",
                                    tag=f"{job_name}-{fname}.space",
                                    num_img=f"{per_img_num[3]}",
                                    config_file=f"{conf}",
                                    corpus_dir=f"{tmp_prefix}{corpus_f}",
                                    fonts_list="data/fonts_list/chn.txt",
                                    corpus_mode="list",
                                    output_dir=f"{tmp_prefix}{self.o_dir}")
                        x3['config_file'] = 'configs/mix_data_space.yaml'
                        x3['img_width'] = tmp_w
                        x3['img_height'] = tmp_h
                        self.configs.append((x3, True))

                    x4 = dict(strict="",
                                tag=f"{job_name}-{fname}.bg",
                                num_img=f"{per_img_num[4]}",
                                config_file=f"{conf}",
                                corpus_dir=f"{tmp_prefix}{corpus_f}",
                                fonts_list="data/fonts_list/chn.txt",
                                corpus_mode="list",
                                output_dir=f"{tmp_prefix}{self.o_dir}")
                    x4['config_file'] = 'configs/mix_data_bg.yaml'
                    x4['bg_dir'] = f"{tmp_prefix}data/bg_base"
                    x4['img_width'] = tmp_w
                    x4['img_height'] = tmp_h
                    self.configs.append((x4, True))

                    if "   " in content:
                        sf.write(f"configs/mix_data_mix.yaml-content three space==={line}\n")
                    else:
                        x5 = dict(strict="",
                                    tag=f"{job_name}-{fname}.basemix",
                                    num_img=f"{per_img_num[5]}",
                                    config_file=f"{conf}",
                                    corpus_dir=f"{tmp_prefix}{corpus_f}",
                                    fonts_list="data/fonts_list/chn.txt",
                                    corpus_mode="list",
                                    output_dir=f"{tmp_prefix}{self.o_dir}")
                        x5['config_file'] = 'configs/mix_data_mix.yaml'
                        x5['bg_dir'] = f"{tmp_prefix}data/bg_base"
                        x5['img_width'] = tmp_w
                        x5['img_height'] = tmp_h
                        self.configs.append((x5, True))

                    if "   " in content:
                        sf.write(f"configs/mix_data_customer.yaml-content three space==={line}\n")
                    else:
                        x6 = dict(strict="",
                                    tag=f"{job_name}-{fname}.customer",
                                    num_img=f"{per_img_num[6]}",
                                    config_file=f"{conf}",
                                    corpus_dir=f"{corpus_f}",
                                    corpus_mode="list",
                                    output_dir=f"{self.o_dir}")
                        x6['config_file'] = 'configs/mix_data_customer.yaml'
                        x6['bg_dir'] = f"data/bg_base"
                        x6['img_width'] = tmp_w
                        x6['img_height'] = tmp_h
                        self.configs.append((x6, False))
                else:
                    sf.write(f"{line}\n")


    def __init__(self, image_dir_path="", train_file="",
    per_img_num=(96, 32, 256), conf="configs/mix_data_base.yaml",
    corpus_dir="data/list_corpus", o_dir="output/mix_train",
    key_file="",
    job_name="collect_texture",
    is_fix=False, have_img=True, list_corpus="", texture_pkl="data/base_texture.pkl"
    ):
        # vim -d configs/default.yaml configs/mix_data.yaml
        # mkdir caonima; cd caonima; git clone ; checkout
        tmp_prefix = "../../"

        self.is_fix = is_fix
        self.job_name = job_name
        self.key_file = key_file

        self.widthList = []
        self.heightList = []
        self.labelLenList = []

        self.imgdirlist = []
        self.imgdirlist.append(image_dir_path)
        self.o_dir = o_dir + "_" + self.job_name
        self.configs = []
        self.filelist = []
        self.textureList = []
        self.filelist.append(train_file)

        os.makedirs(f"data/bg_base", exist_ok=True)
        # 如果不是修补。。
        if not is_fix:
            self.__base_have_image__(image_dir_path, train_file, corpus_dir, per_img_num, conf, tmp_prefix, job_name, is_fix)

            self.df = pd.DataFrame(np.array(self.textureList), columns=['width', 'height', 'char_distribute', 'bg_store'])
            subprocess.getoutput(f"rm -rf data/{job_name}-base_texture.pkl")
            self.df.to_pickle(f"data/{job_name}-base_texture.pkl")

            self.__getUniSize__(is_fix)
            print(f"{job_name}----uni_h={self.uni_h}-----uni_w={self.uni_w}--height,O_width, uni_width, labelLen--mean, stddev, min, max , median, stdrate, 2stdrate, 3stdrate--the distribution is{self.result} ")
        elif have_img:
            # 是fix修补。但是有图的情况
            self.__base_have_image__(image_dir_path, train_file, corpus_dir, per_img_num, conf, tmp_prefix, job_name, is_fix)
            self.__getUniSize__(is_fix)

        else:
            # 是fix修补，但是没有图，只得加载textureList进行生产。导入语料被定义为是修补，就是is_fix=True。
            # 第一步加载纹理框
            self.dfTexture = pd.read_pickle(texture_pkl)
            # 找到逻辑了，通过长度，找所有长度为这个长度的语料。然后，开始搞。所以，这里就不是写一行了, 语料里面是很多行。
            # 所有没有匹配到的语料。最终拼接成一行。就行顺序截取。一一对应。。while（true），直到消耗殆尽。。
            
            pass

    def __counter__(self, i, a, b):
        counter = 0
        for e in i:
            if e > math.floor(a) and e < math.floor(b):
                counter += 1
        return counter

    def getStat(self, i):
        # i是待评估的list。 按照顺序存放，mean, 标准差，
        result = []
        a = statistics.mean(i)
        b = statistics.stdev(i)
        size = len(i)/1.0
        result.append(a)
        result.append(b)
        result.append(min(i))
        result.append(max(i))
        result.append(statistics.median(i))
        #统计一阶标准差，二阶标准差，三阶标准差占比
        result.append(self.__counter__(i, a - b,  a + b) / size)
        result.append(self.__counter__(i, a - 2*b,  a + 2*b) / size)
        result.append(self.__counter__(i, a - 3*b,  a + 3*b) / size)                       
        return result

    def __getUniSize__(self, is_fix, unih=None, uniw=None):
        # 去统计图片的数据情况。主要是高和宽。。返回的是，统一以后的长度。。
        # 这个结果，主要是用于resize图片。。
        # 第一步得到height的分布情况。。
        self.result = []
        if is_fix:
            if unih is not None:
                self.uni_h = unih
            else:
                self.uni_h = 158
            if uniw is not None:
                self.uni_w = uniw
            else:
                self.uni_w = 686
        else:
            t_height_stat = self.getStat(self.heightList)
            self.result.append(t_height_stat)
            o_width_stat = self.getStat(self.widthList)
            self.result.append(o_width_stat)
            a_mean = t_height_stat[0]
            self.uni_h = 158
            a_stddev = t_height_stat[1]
            if len(self.heightList) > 4096:
                self.uni_h = math.floor(a_mean + 2.0 * a_stddev)
            # 第二步，统计归一化以后的情况。。
            tmp_w_list = []
            for e, f in zip(self.widthList, self.heightList):
                t_scale = f * 1.0 / self.uni_h
                tmp_w_list.append(math.floor(e * 1.0 / t_scale))
            t_width_stat = self.getStat(tmp_w_list)
            self.result.append(t_width_stat)
            b_mean = t_width_stat[0]
            self.uni_w = 686
            b_stddev = t_width_stat[1]
            if len(self.widthList) > 4096:
                self.uni_w = math.floor(b_mean + 2.0 * b_stddev)
            # 统计长度的结果
            o_label_len = self.getStat(self.labelLenList)
            self.result.append(o_label_len)

    def __dict_to_args__(self, config: dict):
        args = []
        for k, v in config.items():
            if v is False:
                continue
            args.append('--%s' % k)
            args.append('%s' % v)
        return args
    def __p_process__(self, cmd, params):
        os.system(cmd + " " + " ".join(params))

    def run_gen(self, pool_len=1):
        # self.main_func = './main.py'
        # 先做一些清理工作。
        if os.path.exists(self.o_dir):
            shutil.rmtree(self.o_dir)
        # 对于base的东西使用shell进行调用。。
        pool = Pool(pool_len)
        for config, flag in self.configs:
            xargs = self.__dict_to_args__(config)
            # print("Run with args: %s" % xargs)
            if flag:
                # subprocess.run(['sh', "exe_original.sh"] + [" ".join([str(e) for e in xargs])])
                pool.apply_async(self.__p_process__, args=("sh exe_original.sh", xargs))
            else:
                # subprocess.run(['python', self.main_func] + xargs)
                pool.apply_async(self.__p_process__, args=('python ./main.py', xargs))
            # 务必休息100s否则，机器一定会崩掉。
            # time.sleep(100)
        pool.close()
        pool.join()
            
    def merge_result(self, out_suffix="_result"):
        self.out = self.o_dir + out_suffix
        if os.path.exists(self.out):
            shutil.rmtree(self.out)
        os.mkdir(self.out)
        with open(os.path.join(self.out, "tmp_labels.txt"), "w", encoding='utf-8') as resultf:
            for dir_path, dir_name_list, file_name_list in os.walk(self.o_dir):
                if dir_path != self.o_dir:
                    # basename现在是路径的最后一个。
                    basename = dir_path.split("/")[-1]
                    # 如果不是隐藏文件。
                    if not basename.startswith("."):
                        # basename用点.做分割；得到.jpg前面的部分；然后继续用点.得到类型。customer，noline，或者line。。
                        # 这里面就要求jobname，不能含有点.号。
                        basename = basename.split(".")[0] + "_" + basename.split(".")[-1] + "_"
                        with open(os.path.join(dir_path, "tmp_labels.txt"), encoding='utf-8') as f:
                            tmp_flist = [str(line).strip() for line in f.readlines()]
                        for line in tmp_flist:
                            fname = line.split(" ")[0]
                            content = line[len(fname):]
                            # 数据两边是不允许加空格的
            #                 content = content.strip()
                            fname = fname + ".jpg"
                            fname_path = os.path.join(dir_path, fname)
                            resultf.write(f"{basename + fname}{content}\n")
                            shutil.copy2(fname_path, os.path.join(self.out, basename + fname))
        self.filelist.append(os.path.join(self.out, "tmp_labels.txt"))
        # 在merge的时候，存在字典的问题。。。
        self.__create__()
        self.__getIndex__()
        self.imgdirlist.append(self.out)

    def __create__(self):
        # 如果是修补，那么字典就是确定的，不存在生产新的字典。。
        if not self.is_fix:
            self.diclist = []
            for file in self.filelist:
                with open(file, encoding='utf-8') as f:
                    for line in f:
                        fname = line.split(" ")[0]
                        tmp = line[len(fname):]
                        tmp = tmp.strip().replace(" ","").replace("卍","")
                        # 在这里做全角转化为半角的转化
                        tmp = tmp.replace("（", "(").replace("）", ")").replace("，", ",")
                        self.diclist.extend(tmp)
            self.counter = Counter(self.diclist)
            with open(os.path.join(self.out, "word_distribution.txt"), "w", encoding='utf-8') as wdf:
                wdf.write(f"--the distribution of the word is {self.counter}")
            # print(f"--the distribution of the word is {self.counter}")
            # 把counter转化为字典，存储起来。
            pkl_wd = open("word_distribution.pkl", "wb")
            pickle.dump(self.counter, pkl_wd)
            self.keys = [' '] + sorted(list(self.counter))
            with open(os.path.join(self.out, "keys.txt"), "w", encoding='utf-8') as kf:
                for i in self.keys:
                    kf.write(i + "\n")
        else:
            # 如果是修补的话，需要传入，keys所在位置。。
            with open(self.key_file, encoding="utf-8") as kf:
                self.keys = [str(e).strip() for e in kf.readlines()]


    def __getIndex__(self):
        for file in self.filelist:
            basename = os.path.basename(file)
            with open(file, encoding='utf-8') as f, open(file + "_with_index.txt", "w", encoding='utf-8') as indf:
#             with open(file) as f, open(os.path.join(self.out, basename + "_with_index"), "w") as indf:                
                for line in f:
                    fname = line.split(" ")[0]
                    content = line[len(fname):]
                    content = content.strip().replace(" ","").replace("卍","")
                    # 在这里做全角半角的转化。。
                    content = content.replace("（", "(").replace("）", ")").replace("，", ",")
                    indf.write(fname)
                    for e in content:
                        if e != " ":
                            indf.write(" " + str(self.keys.index(e)))
                    indf.write("\n")

    def resizeImg(self, result_suffix="_fixresize"):
        # 预测函数，要强行转化为32，686
        print(f"----image directory is {self.imgdirlist}---")
        for img_dir in self.imgdirlist:            
            finout = img_dir + result_suffix
            if os.path.exists(finout):
                shutil.rmtree(finout)
            os.mkdir(finout)
            # 遍历所有的图片，然后进行resize。。
            for file in glob.glob(f"{img_dir}/*.*"):
                filename = os.path.basename(file)
                if '.jpg' in file or '.jpeg' in file:
                    # 打开图片然后resize就好。。
                    img = Image.open(file)
                    img = img.resize((self.uni_w,self.uni_h), Image.ANTIALIAS)
                    img.convert('RGB').save(os.path.join(finout, filename))
                else:
                    shutil.copy2(file, finout)
            
        
def resizeImg(unih=158, uniw=686, result_suffix="_fixresize"):
    # 预测函数，要强行转化为32，686
    imgdirlist = ['/workspace/densent_ocr/only_qishui_debug', 'output/only_debug_keys1_mergekeys_1']
    for img_dir in imgdirlist:
        finout = img_dir + result_suffix
        if os.path.exists(finout):
            shutil.rmtree(finout)
        os.mkdir(finout)
        # 遍历所有的图片，然后进行resize。。
        for file in glob.glob(f"{img_dir}/*.*"):
            filename = os.path.basename(file)
            if '.jpg' in file or '.jpeg' in file:
                # 打开图片然后resize就好。。
                img = Image.open(file)
                img = img.resize((uniw,unih), Image.ANTIALIAS)
                img.convert('RGB').save(os.path.join(finout, filename))
            else:
                shutil.copy2(file, finout)

def fix_keys_index(fix_label_file_l=None, merge_file_l=None, out="."):
    #image_name.jp*g<空格>content
    # fix_label_file_l 和 merge_file_l是两个list。
    filelist = fix_label_file_l + merge_file_l
    diclist = []
    for file in filelist:
        with open(file, encoding='utf-8') as f:
            for line in f:
                fname = line.split(" ")[0]
                tmp = line[len(fname):]
                tmp = tmp.strip().replace(" ","").replace("卍","")
                # 在这里做全角转化为半角的转化
                tmp = tmp.replace("（", "(").replace("）", ")").replace("，", ",")
                diclist.extend(tmp)
	#	输出字频分布情况
    counter = Counter(diclist)
    print(f"--the distribution of the word is {counter}")
    # 把counter转化为字典，存储起来。
    keys = [' '] + sorted(list(counter))
    with open(os.path.join(out, "new_keys.txt"), "w", encoding='utf-8') as kf:
        for i in keys:
            kf.write(i + "\n")
    # 接下来就是进行，字典转化的问题了。。。
    for file in filelist:
        basename = os.path.basename(file)
        with open(file, encoding='utf-8') as f, open(file + "_with_new_index.txt", "w", encoding='utf-8') as indf:
        #             with open(file) as f, open(os.path.join(out, basename + "_with_index"), "w") as indf:
            for line in f:
                fname = line.split(" ")[0]
                content = line[len(fname):]
                content = content.strip().replace(" ","").replace("卍","")
                # 在这里做全角半角的转化。。
                content = content.replace("（", "(").replace("）", ")").replace("，", ",")
                indf.write(fname)
                for e in content:
                    if e != " ":
                        indf.write(" " + str(keys.index(e)))
                indf.write("\n")


# 修补程序测试通过。
# x = gexinghuaRunner(image_dir_path="/Users/GuoLiuFang/Downloads/only_qishui_stdard",
# train_file="/Users/GuoLiuFang/Downloads/label_tmp_all20190311.txt_filter_l.txt",
# o_dir="output/test_fix",
# per_img_num=(1, 2, 3, 4, 5, 6, 7),
# job_name="test_fix_job_name",
# is_fix=True,
# have_img=True,
# key_file="/Users/GuoLiuFang/Downloads/keys.txt"
# )
# # 测试crate程序当前的现状
# x = gexinghuaRunner(image_dir_path="/Users/GuoLiuFang/Downloads/only_qishui_stdard",
# train_file="/Users/GuoLiuFang/Downloads/label_tmp_all20190311.txt_filter_l.txt",
# o_dir="output/test_fix",
# # per_img_num=(100, 32, 32, 32, 32, 72, 100),
# per_img_num=(1, 2, 3, 4, 5, 6, 7),
# # per_img_num=(100, 32, 32, 32, 32, 72, 100), 每张图400张。
# job_name="test_create_job_name",
# is_fix=False,
# key_file="/Users/GuoLiuFang/Downloads/keys.txt"
# )

x = gexinghuaRunner(image_dir_path="/workspace/densent_ocr/become_legend",
train_file="/workspace/densent_ocr/become_legend/become_legend_finnaly.txt",
o_dir="output/dare_to_life",
per_img_num=(128, 16, 16, 16, 16, 64, 64),
#per_img_num=(1, 2, 3, 4, 5, 6, 7),
# per_img_num=(100, 32, 32, 32, 32, 72, 100), 每张图400张。
job_name="you_are_the_legend",
is_fix=False,
key_file=""
)


# x = gexinghuaRunner(image_dir_path="/workspace/densent_ocr/only_qishui_stdard",
# train_file="/workspace/densent_ocr/only_qishui_stdard/label_tmp_all20190311.txt",
# o_dir="output/only_all"
# )
x.run_gen(pool_len=14)
x.merge_result(out_suffix="_merge")
x.resizeImg(result_suffix="_resize")
# resizeImg(result_suffix="_keys_acsii")

import os
import subprocess
from PIL import Image
import shutil
from collections import Counter
import glob
import pandas as pd
import numpy as np
class gexinghuaRunner:
    def __init__(self, image_dir_path="", train_file="", per_img_num=64, conf="configs/mix_data.yaml", corpus_dir="data/list_corpus", o_dir="output/mix_train"):
        self.imgdirlist = []
        self.imgdirlist.append(image_dir_path)
        self.o_dir = o_dir
        self.configs = []
        self.filelist = []
        self.textureList = []
        self.filelist.append(train_file)
        with open(os.path.join(image_dir_path, train_file), encoding='utf-8') as f:
            test_image_list = [str(line).strip() for line in f.readlines()]
        for line in test_image_list:
            fname = line.split(" ")[0]
            content = line[len(fname):]
            content = content.strip()
            corpus_f = os.path.join(corpus_dir, fname)
            if os.path.exists(corpus_f):
                shutil.rmtree(corpus_f)
            os.mkdir(corpus_f)
            with open(f"{corpus_f}/{fname}.txt", "w", encoding='utf-8') as tmpf:
                tmpf.write(f"{content}\n")
            tmpdict = dict(strict="", 
                           tag=f"{fname}", 
                           num_img=f"{per_img_num}", 
                           config_file=f"{conf}", 
                          corpus_dir=f"{corpus_f}", 
                          fonts_list="data/fonts_list/base_chn.txt",
                          corpus_mode="list", 
                          output_dir=f"{self.o_dir}")
            tmpimg = Image.open(os.path.join(image_dir_path, fname))
            tmp_w = tmpimg.size[0]
            tmp_h = tmpimg.size[1]
            tmpdict['img_width'] = tmp_w
            tmpdict['img_height'] = tmp_h
            self.configs.append(tmpdict)
            ### 新增纹理信息收集texture..以及bg的。。
#             parser.add_argument('--bg_dir', type=str, default='./data/bg',
#                         help="Some text images(according to your config in yaml file) will"
#                              "use pictures in this folder as background")            
            # 在这里新建./data/bg_base/fname/然后这里存放内容就好。
            if os.path.exists(f"data/bg_base/{fname}"):
                shutil.rmtree(f"data/bg_base/{fname}")
            os.makedirs(f"data/bg_base/{fname}", exist_ok=True)
            tmpimg.crop([0, 0, tmp_w * 0.05 + 1, tmp_h]).convert('RGB').save(f"data/bg_base/{fname}/1.jpg")
            tmpimg.crop([0, 0, tmp_w, tmp_h * 0.05 + 1]).convert('RGB').save(f"data/bg_base/{fname}/2.jpg")
            tmpimg.crop([tmp_w * 0.95 - 1, 0, tmp_w, tmp_h]).convert('RGB').save(f"data/bg_base/{fname}/3.jpg")
            tmpimg.crop([0,tmp_h * 0.95 - 1, tmp_w , tmp_h]).convert('RGB').save(f"data/bg_base/{fname}/4.jpg")
            self.textureList.append([tmp_w, tmp_h, content, f"data/bg_base/{fname}"])
            ### table line 和random space个16张。。bg和blur个8张。。。
            x1 = dict(strict="", 
                           tag=f"{fname}", 
                           num_img=f"{per_img_num}", 
                           config_file=f"{conf}", 
                          corpus_dir=f"{corpus_f}", 
                          fonts_list="data/fonts_list/base_chn.txt",
                          corpus_mode="list", 
                          output_dir=f"{self.o_dir}")
            x1['config_file'] = 'configs/mix_data_line.yaml'
            x1['num_img'] = 16
            self.configs.append(x1)
            # 判断是否含有三个空格，如果
            if "   " not in content:
                x2 = dict(strict="", 
                           tag=f"{fname}", 
                           num_img=f"{per_img_num}", 
                           config_file=f"{conf}", 
                          corpus_dir=f"{corpus_f}", 
                          fonts_list="data/fonts_list/base_chn.txt",
                          corpus_mode="list", 
                          output_dir=f"{self.o_dir}")
                x2['config_file'] = 'configs/mix_data_space.yaml'   
                x2['num_img'] = 16
                self.configs.append(x2)
            # 接下来是添加bg和blur，各8张图片。。
            x3 = dict(strict="", 
                           tag=f"{fname}", 
                           num_img=f"{per_img_num}", 
                           config_file=f"{conf}", 
                          corpus_dir=f"{corpus_f}", 
                          fonts_list="data/fonts_list/base_chn.txt",
                          corpus_mode="list", 
                          output_dir=f"{self.o_dir}")
            x3['config_file'] = 'configs/mix_data_bg.yaml'
            x3['num_img'] = 16
            x3['bg_dir'] = f"data/bg_base/{fname}"
            self.configs.append(x3)
            # blur 也是8张图片
            x4 = dict(strict="", 
                           tag=f"{fname}", 
                           num_img=f"{per_img_num}", 
                           config_file=f"{conf}", 
                          corpus_dir=f"{corpus_f}", 
                          fonts_list="data/fonts_list/base_chn.txt",
                          corpus_mode="list", 
                          output_dir=f"{self.o_dir}")
            x4['config_file'] = 'configs/mix_data_blur.yaml'
            x4['num_img'] = 16           
            self.configs.append(x4)
            # 重型mix_mix
            x5 = dict(strict="", 
                           tag=f"{fname}", 
                           num_img=f"{per_img_num}", 
                           config_file=f"{conf}", 
                          corpus_dir=f"{corpus_f}", 
                          corpus_mode="list", 
                          output_dir=f"{self.o_dir}")
            x5['config_file'] = 'configs/mix_data_mix.yaml'
            x5['num_img'] = 128         
            self.configs.append(x5)            
        # 把纹理特征存储起来以供后面使用。。注意程序的border。。
        self.df = pd.DataFrame(np.array(self.textureList), columns=['width', 'height', 'char_distribute', 'bg_store'])     
        subprocess.getoutput("rm -rf data/base_texture.pkl")
        self.df.to_pickle("data/base_texture.pkl")
    def __dict_to_args__(self, config: dict):
        args = []
        for k, v in config.items():
            if v is False:
                continue
            args.append('--%s' % k)
            args.append('%s' % v)
        return args
    def run_gen(self):
        self.main_func = './main.py'
        # 先做一些清理工作。
        if os.path.exists(self.o_dir):
            shutil.rmtree(self.o_dir)
        for config in self.configs:
            args = self.__dict_to_args__(config)
            print("Run with args: %s" % args)
            subprocess.run(['python', self.main_func] + args)
    def merge_result(self, out_suffix="_result"):
        self.out = self.o_dir + out_suffix
        if os.path.exists(self.out):
            shutil.rmtree(self.out)
        os.mkdir(self.out)
        with open(os.path.join(self.out, "tmp_labels.txt"), "w", encoding='utf-8') as resultf:
            for dir_path, dir_name_list, file_name_list in os.walk(self.o_dir):
                if dir_path != self.o_dir:
    #                 print(dir_path)
                    # 读取文件内容，然后进行复制操作。
                    basename = dir_path.split("/")[-1]
                    if not basename.startswith("."):
                        basename = basename.split(".")[0] + "_"
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
        self.__create__()
        self.__getIndex__()
        self.imgdirlist.append(self.out)
    def __create__(self):
        self.diclist = []
        for file in self.filelist:
            with open(file, encoding='utf-8') as f:
                for line in f:
                    fname = line.split(" ")[0]
                    tmp = line[len(fname):]
                    tmp = tmp.strip().replace(" ","")
                    # 在这里做全角转化为半角的转化
                    tmp = tmp.replace("（", "(").replace("）", ")").replace("，", ",")
                    self.diclist.extend(tmp)
        self.counter = Counter(self.diclist)
        # 把counter转化为字典，存储起来。
        self.keys = [' '] + sorted(list(self.counter))
        with open(os.path.join(self.out, "keys.txt"), "w", encoding='utf-8') as kf:
            for i in self.keys:
                kf.write(i + "\n")
    def __getIndex__(self):
        for file in self.filelist:
            basename = os.path.basename(file)
            with open(file, encoding='utf-8') as f, open(file + "_with_index.txt", "w", encoding='utf-8') as indf:
#             with open(file) as f, open(os.path.join(self.out, basename + "_with_index"), "w") as indf:                
                for line in f:
                    fname = line.split(" ")[0]
                    content = line[len(fname):]
                    content = content.strip().replace(" ","")
                    # 在这里做全角半角的转化。。
                    content = content.replace("（", "(").replace("）", ")").replace("，", ",")
                    indf.write(fname)
                    for e in content:
                        if e != " ":
                            indf.write(" " + str(self.keys.index(e)))
                    indf.write("\n")
#     def getUniSize(self):
        # 去统计图片的数据情况。主要是高和宽。。返回的是，统一以后的长度。。
    def resizeImg(self, unih=32, uniw=686, result_suffix="_fixresize"):
        # 预测函数，要强行转化为32，686
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
                    img = img.resize((uniw,unih), Image.ANTIALIAS)
                    img.convert('RGB').save(os.path.join(finout, filename))
                else:
                    shutil.copy2(file, finout)
            
        
def resizeImg(unih=158, uniw=686, result_suffix="_fixresize"):
    # 预测函数，要强行转化为32，686
    imgdirlist = ['/workspace/densent_ocr/only_qishui', 'output/only_qishui_final_glf_result_1']
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











#x = gexinghuaRunner(image_dir_path="/workspace/densent_ocr/only_qishui",
#train_file="/workspace/densent_ocr/only_qishui/label_tmp_guaid_data_produce.txt",
#o_dir="output/only_qishui_final"
#)
#x.run_gen()
#x.merge_result(out_suffix="_glf_result_1")
resizeImg(result_suffix="_rz_fixresize_1")

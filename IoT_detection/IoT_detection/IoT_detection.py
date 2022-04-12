import pickle
import re
import glob
import argparse
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

model_path = "./logistic/"

def parameter_parse():
    parser = argparse.ArgumentParser(description = "IoT Detecting...")
    parser.add_argument("-f", nargs = "?", help = "Input file with .log or .txt format.")

    return parser.parse_args()


class Iot_detection:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.tfidf_data = self.tfidf(data)
        self.ismal = self.ware_type(self.model, self.tfidf_data)

    def tfidf(self, data):
        #print(data)
        with open(model_path + 'vector.pkl', 'rb') as f:
            vec = pickle.load(f)
        with open(model_path + 'tfidf.pkl', 'rb') as f:
            tfidf_machine = pickle.load(f)
        
        data = vec.transform(data)

        data = tfidf_machine.transform(data)

        return data  

    def ware_type(self, model, data):
        benign = 0
        mal = 0

        for m in model:
            ans = int(m.predict(data)[0])
            
            if ans == 0:
                benign += 1
            else:
                mal += 1

        if benign > mal:
            return False
        else:
            return True

def get_log(path):
    tooshort = False
    log = []

    if os.path.getsize(path) <= 1024:
        tooshort = True
    else:
        
        f = open(path, 'r', encoding = 'utf-8')
    
        for l in f.readlines():
            log.append(l)
        
    return log, tooshort

def extract_feature(log):
    pdata = []
    cmd = ""
    process_attach = False
    pattern = re.compile('(\s)+=(\s)+(-)*(\d)+')
    ques_pattern = re.compile('(\s)+=(\s)+?')
    ignore_list = ["No such process", "timeout", "Operation not permitted"
                   , "Invalid argument", "<detached ...>"]

    
    for line in log:
        if process_attach:
            ignore = False

            for i in ignore_list:
                if i in line and len(cmd) != 0:
                    ignore = True
                    break
                    
            if not ignore:         
                if line.find("strace: Process") != -1 and line.find("detached") != -1:
                    pdata.append(cmd)
                    process_attach = False
                
                elif re.search(pattern, line) != None:
                    start = line.find(" ")
                    end = line.find(re.search(pattern, line)[0])
                    cmd += line[start : end]

                elif re.search(ques_pattern, line) != None:
                    start = line.find(" ")
                    end = line.find(re.search(ques_pattern, line)[0])
                    cmd += line[start : end]

                elif line.find("+++") != -1:
                    start = line.find(" ")
                    cmd += line[start:]
                    pdata.append(cmd)
                    process_attach = False

            ignore = False

        else:
            if line.find('[SCAN_START]') != -1:
                process_attach = True

    if len(pdata) == 0:
        pdata.append(cmd)
    return pdata

def read_model_data(path):
    model = []

    for i in range(1, 6):
        with open(model_path + f'model{i}.pkl', 'rb') as f:
            model.append(pickle.load(f))
    
    log, tooshort = get_log(path)
    data = []

    if not tooshort:
        data = extract_feature(log)

    return model, data, tooshort
    
def main(path):
    model, data, tooshort = read_model_data(path)

    if tooshort:
        return False
    else:
        iot = Iot_detection(model, data)
        return iot.ismal

if __name__ == '__main__':
    #args = parameter_parse()

    path = glob.glob("./infected-routerlog_nodelete/*/*.log")
    file = 0
    true = 0
    false = 0

    for p in path:
        file += 1
        try:
            if main(p):
                true += 1
            else:
                false += 1
        except:
            print(p)

    print(file)
    print(true)
    print(false)
    

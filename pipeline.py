import sys
from sys import argv
from Experiment import *
import pandas as pd
from sklearn.kernel_approximation import Nystroem
import os
from tools import *



def experiment_shortTS(method,outputfile):
    mod = sys.modules["__main__"]

    func = getattr(mod, method)

    #parameter
    all_para=pd.read_csv("parameter/pipeline.csv")
    data_list=np.array(all_para['name'])
    cycle_list=np.array(all_para['cycle'])
    str_list=np.array(all_para['anomaly cycle'])
    ano_list=[]
    for st in str_list:
        ano_list.append(np.array(st.split(','),dtype=int))
    folder = os.path.exists(outputfile)
    if folder:
        print("folder exists")
        return
    else:
        os.mkdir(outputfile)
    #遍历data_list
    result=np.full(len(data_list),-1.0)
    for i in range(len(data_list)):
        result[i]=func(data_list[i],cycle_list[i],ano_list[i],outputfile)
    name = ['data', 'result']
    test = pd.DataFrame(columns=name, data=np.vstack((data_list, result)).T)
    output_file = outputfile+ '/result.csv'
    test.to_csv(output_file, encoding='gbk')





if __name__ == '__main__':
    experiment_shortTS(method=argv[1],outputfile=argv[2])




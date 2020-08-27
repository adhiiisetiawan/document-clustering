import pandas as pd
import numpy as np

def tfIdfCalculation(dataset):    
    # Remove duplicates value
    string = " "
    data = string.join(dataset)
    
    tokens = list(dict.fromkeys(data.split()))
    
    doc_fr = []
    for d in dataset:
        tmp = []
        for t in tokens:
            tmp.append([t, d.split().count(t)])
        doc_fr.append(tmp)
    
    c = 1
    for tf in doc_fr:
        print("doc", c)
        c += 1
        i = 0
        for h in tf:
            if(h[1] != 0):
                tmp = list(h)
                tmp[1] = np.log10(h[1])
                tf[i] = tmp
            print(tf[i][0], "\t\t= ", tf[i][1])
            i += 1
        print("=========================")
    print(doc_fr)
#    # Create the pandas DataFrame 
#    df = pd.DataFrame(doc_fr, columns = ["doc 1", "doc 2", "doc 3"]) 
#  
#    # print dataframe. 
#    print(df)
    return
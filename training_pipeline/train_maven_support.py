import numpy as np

# This function takes a list and write to a file. Each item in a list is separated by a new line
# LIST = list, list of items to write to file
# FILENAME = str, filename (ends in .txt)
def list_to_file(LIST,FILENAME):
    openfile= open(FILENAME,'w')
    for item in LIST:
        openfile.write(str(item)+'\n')
    openfile.close()

# This function reads a file into a list (Each item is separated by a new line)
# FILENAME = str, filename (ends in .txt)
def file_to_list(FILENAME):
    l = []
    openfile= open(FILENAME,'r')
    for line in openfile.readlines():
        line = line.strip()
        l.append(line)
    openfile.close()
    return(l)

# Average predictions from ensembl model   
def predict_ensembl(models,data):
    all_predictions = [] 
    for model in models:
        all_predictions.append([x[1] for x in model.predict_proba(data)])
    all_predictions = np.mean(all_predictions,axis=0)
    return(all_predictions)

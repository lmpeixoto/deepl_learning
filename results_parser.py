import collections
import os
import re
import pandas as pd

list_models = ['svm', 'lda', 'knn', 'rf', 'lr']
list_features = [100, 500, 1000, 2000, 5000, 10000]
list_endpoints = ['endpoint_Sex', 'endpoint_EFSAll', 'endpoint_ClassLabel', 'endpoint_OSAll', 'endpoint_OSHR',
                      'endpoint_EFSHR']

nested_dict = lambda: collections.defaultdict(nested_dict)

results = nested_dict()
root_dir = 'Shallow_Model_Selection_Results_original'
for endpoint in list_endpoints:
    for features in list_features:
        for model in list_models:
            file_name = model + '_' + str(features) + '_' + 'report_0.txt'
            print(file_name)
            file_path = os.path.join(root_dir, endpoint, file_name)
            f = open(file_path, 'r')
            text = f.readlines()
            text = text[-7:]
            for line in text:
                score = re.split(':|\n', line)
                if score[0] == 'mcc':
                    results[endpoint][model][features] = float(score[1])
            f.close()

#convert defaultdict to dict
results = dict(results)
for endpoint in list_endpoints:
    results[endpoint] = dict(results[endpoint])
    for model in list_models:
        results[endpoint][model] = dict(results[endpoint][model])


writer = pd.ExcelWriter('results_shallow_original_split.xlsx')
for endpoint in list_endpoints:
    df = pd.DataFrame.from_dict(results[endpoint]).T
    df.to_excel(writer, sheet_name=endpoint)
writer.save()






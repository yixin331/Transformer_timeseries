import csv
import json
from statistics import mean, stdev
import numpy as np

labelPath = '../data_new/test/testSummaryLabel_3.txt'
goldPath = '../data_new/test/testOriginalSummary_3.txt'
summaryPath = '../data_new/test/testSummary_3.txt'
IDPath = '../data_new/test/testID_3.txt'

generatedPath = '../results_new/helen3/generated/'

fillers = ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', ':', '-',
           'to', 'but', 'is', 'of', 'it', 'on', '.', 'at', '(', ')', ',', ';']

count = 1

generatedScores = []


with open(labelPath, 'r', encoding='utf-8') as labelFile, open(summaryPath, 'r', encoding='utf-8') as summaryFile, \
        open(goldPath, 'r', encoding='utf-8') as goldFile, open(IDPath, 'r', encoding='utf-8') as IDFile:
    for lbls, summary, gold in zip(labelFile.readlines(), summaryFile.readlines(), goldFile.readlines()):
        if count in [2, 12, 77]:
            print(count)
        labelArr = lbls.split()
        summArr = summary.split()
        goldArr = gold.split()
        recordList = []
        for lab, sums, gld in zip(labelArr, summArr, goldArr):
            if lab == '1' and gld.lower() not in fillers and gld.lower() not in recordList:
                recordList.append(gld.lower())
        list1 = recordList
        list2 = recordList
        list3 = recordList
        recordLength = len(recordList)
        generatedList = []
        with open(generatedPath + str(count) + '.json') as generatedFile:
            document1 = json.loads(generatedFile.read())
            summary1 = ''.join(document1['summary'])
        for token in summary1.split():
            if token.lower() in list1:
                list1.remove(token.lower())
                generatedList.append(token.lower())

        count += 1
        if recordLength == 0:
            generatedRatio = 0
        else:
            generatedRatio = len(generatedList) / recordLength

IDfile = open(IDPath, 'r')
IDs = [line for line in IDfile.read().splitlines()]
unique_IDs = np.unique(np.array(IDs))
best_generatedScores = []
avg_generatedScores = []
for i,  unique_ID in enumerate(unique_IDs):
    indices = [i for i in range(len(IDs)) if IDs[i] == unique_ID]
    best_generatedScores.append(np.max([generatedScores[i] for i in indices]))
    avg_generatedScores.append(np.average([generatedScores[i] for i in indices]))

print(f'generated best CS stdev: {round(stdev(best_generatedScores)*100,2)}%')
print()
print(f'generated best CS mean: {round(mean(best_generatedScores)*100,2)}%')
print()
print(f'generated best CS RSD: {round((stdev(best_generatedScores)*100) / abs(mean(best_generatedScores)),2)}%')
print()
print(f'generated avg CS stdev: {round(stdev(avg_generatedScores)*100,2)}%')
print()
print(f'generated avg CS mean: {round(mean(avg_generatedScores)*100,2)}%')
print()
print(f'generated avg CS RSD: {round((stdev(avg_generatedScores)*100) / abs(mean(avg_generatedScores)),2)}%')

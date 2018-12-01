import csv
import json
from collections import Counter

# data = {}
# with open('review_classification.json','r') as f1:
#     review= json.loads(f1.read())
#     # print(type(review))
#     for k,v in review.items():
#         if k not in data:
#             data[k] = []
#         data[k].append(v)
#         data[k].append(0)
#     print(data)
#
# with open('check_in_count.json','r') as f2:
#     check_in = json.loads(f2.read())
#     # print(type(review))
#     for k,v in check_in.items():
#         if k not in data:
#             data[k] = ['NA',v]
#         else:
#             data[k][1] = v
#     print(data)
#
# file_str = 'business_id, review, checkin\n'
# with open('output.csv','w', newline='') as file:
#     file.write(file_str)
#     writer = csv.writer(file)
#     for k,v in data.items():
#         writer.writerow([k,v[0],v[1]])


#     for row in csv_f1:
#         if row[0] not in data:
#             data[row[0]] = []
#         data[row[0]].append(row[1])
#
# with open('check_in_count.csv','r') as f2:
#     csv_f2 = csv.reader(f2, delimiter=',')
#     for row in csv_f1:
#         if row[0] not in data:
#             data[row[0]] = ['NA']
#         data[row[0]].append(row[1])
#
# print(data)

with open('sentiment_dict.json','r') as f:
    sent = json.loads(f.read())
    # print(sent)
    for k,v in sent.items():
        sent[k] = Counter(v)

for k,v in sent.items():
    if 'pos' not in v:
        sent[k]['pos'] = 0
    if 'neg' not in v:
        sent[k]['neg'] = 0

with open("sent.json", 'w') as f:
    json.dump(sent, f)

data = {}
with open('sent.json','r') as f1:
    review= json.loads(f1.read())
    # print(type(review))
    for k,v in review.items():
        if k not in data:
            data[k] = []
        data[k].append(v['pos'])
        data[k].append(v['neg'])
        data[k].append(0)
    # print(data)

with open('check_in_count.json','r') as f2:
    check_in = json.loads(f2.read())
    # print(type(review))
    for k,v in check_in.items():
        if k not in data:
            data[k] = [0, 0, v]
        else:
            data[k][2] = v
    print(data)

file_str = 'business_id,pos_count,neg_count,checkin_count\n'
with open('output.csv','w', newline='') as file:
    file.write(file_str)
    writer = csv.writer(file)
    for k,v in data.items():
        writer.writerow([k,v[0],v[1],v[2]])
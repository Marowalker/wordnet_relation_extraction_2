import os
from synset_file import numbers_only
from collections import defaultdict


def find_sdp_numbers(id, lines):
    start = 0
    for pos, line in enumerate(lines):
        if id == line:
            start = pos
    count = 0
    for i in range(start + 1, len(lines)):
        if not numbers_only(lines[i]):
            count += 1
        else:
            return count
    return len(lines) - 1 - start


datasets = ['train', 'dev', 'test']


for dataset in datasets:
    sdp_order = defaultdict(list)
    print("Process dataset: " + dataset)
    with open('data/raw_data/sdp_data_acentors.' + dataset + '.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if numbers_only(line):
                # print(find_sdp_numbers(line, lines))
                sdps = lines[lines.index(line) + 1:lines.index(line) + find_sdp_numbers(line, lines) + 1]
                for sdp in sdps:
                    pair = sdp.split()[0]
                    sdp_order[line].append(pair)
    with open('data/raw_data/sdp_seq_acentors.' + dataset + '.txt', 'r') as f1:
        l1 = f1.readlines()
    with open('data/raw_data/sdp_new_seq_acentors.' + dataset + '.txt', 'w') as f2:
        for idx in sdp_order:
            f2.write(idx)
            print(idx)
            sdp_switch = l1[l1.index(idx) + 1:l1.index(idx) + find_sdp_numbers(idx, l1) + 1]
            # print(len(sdp_switch))
            temp = []
            # print(sdp_switch)
            for token in sdp_order[idx]:
                # print(token)
                for s in sdp_switch:
                    if token in s.split():
                        if s not in temp:
                            temp.append(s)
            for tok in temp:
                # print(tok)
                f2.write(tok)
            #     f2.write(sdp)
            # print(idx)
            # print(len(sdp_order[idx]))
            # for i in range(len(sdp_order[idx])):
            #     print(sdp_order[idx][i])




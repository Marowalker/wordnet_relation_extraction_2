import constants as constant
from nltk.corpus import wordnet as wn


def _process_data(name):
    with open(name, 'r') as f:
        raw_data = f.readlines()
    data_words, data_postitions, data_y, data_pos, data_synsets, data_relations, \
    data_directions, identities = parse_raw(raw_data)
    words = []
    positions_1 = []
    positions_2 = []
    labels = []
    poses = []
    synsets = []
    relations = []
    directions = []

    for i in range(len(data_postitions)):
        position_1, position_2 = [], []
        e1 = data_postitions[i][0]
        e2 = data_postitions[i][-1]
        for po in data_postitions[i]:
            position_1.append((po - e1 + constant.MAX_LENGTH) // 5 + 1)
            position_2.append((po - e2 + constant.MAX_LENGTH) // 5 + 1)
        positions_1.append(position_1)
        positions_2.append(position_2)

    for i in range(len(data_words)):
        rs = []
        for r in data_relations[i]:
            print(r)


def parse_raw(raw_data):
    all_words = []
    all_positions = []
    all_relations = []
    all_directions = []
    all_poses = []
    all_labels = []
    all_synsets = []
    all_identities = []
    pmid = ''
    for line in raw_data:
        l = line.strip().split()
        if len(l) == 1:
            pmid = l[0]
            # print()
            # print(pmid)
        else:
            pair = l[0]
            label = l[1]
            # print(pair, label)
            if label:
                joint_sdp = ' '.join(l[2:])
                sdps = joint_sdp.split("-PUNC-")
                for sdp in sdps:
                    # S xuoi
                    nodes = sdp.split()
                    words = []
                    positions = []
                    poses = []
                    synsets = []
                    relations = []
                    directions = []

                    for idx, node in enumerate(nodes):
                        node = node.split('|')
                        # print(node)
                        if idx == 0 or idx == len(nodes) - 1:
                            # print(node)
                            for idx, _node in enumerate(node):
                                word = constant.UNK if _node == '' else _node
                                if idx == 0:
                                    # print(word)
                                    w, p, s = word.split('\\')
                                    # w, p, s = word.rsplit('/', 2)
                                    # print(w, p, s)
                                    p = 'NN' if p == '' else p
                                    s = str(wn.synsets('a')[0].offset()) if s == '' else s
                                    _w, position = w.rsplit('_', 1)
                                    words.append(_w)
                                    positions.append(min(int(position), constant.MAX_LENGTH))
                                    poses.append(p)
                                    synsets.append(s)
                                else:
                                    w = word.split('\\')[0]
                        else:
                            dependency = node[0]
                            r = '(' + dependency[3:]
                            d = dependency[1]
                            r = r.split(':', 1)[0] + ')' if ':' in r else r
                            relations.append(r)
                            directions.append(d)

                    all_words.append(words)
                    all_positions.append(positions)
                    all_relations.append(relations)
                    all_directions.append(directions)
                    all_poses.append(poses)
                    all_synsets.append(synsets)
                    all_labels.append([label])
                    all_identities.append((pmid, pair))
            else:
                print(l)

    return all_words, all_positions, all_labels, all_poses, all_synsets, all_relations, \
           all_directions, all_identities


_process_data('data/raw_data/sdf_data_acentors_hypernyms.dev.txt')

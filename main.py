import pickle

import constants
from data_utils import *
from dataset import Dataset
from evaluate.bc5 import evaluate_bc5
from models.model_cnn import CnnModel
from sklearn.utils import shuffle


def main():
    result_file = open('data/results.txt', 'a')

    freq_ent = load_most_freq_entities()

    if constants.IS_REBUILD == 1:
        print('Build data')
        # Load vocabularies
        # vocab_trees = load_vocab(constants.ALL_TREES)
        vocab_words = load_vocab(constants.ALL_WORDS)
        vocab_poses = load_vocab(constants.ALL_POSES)
        vocab_synsets = load_vocab(constants.ALL_SYNSETS)
        vocab_depends = load_vocab(constants.ALL_DEPENDS)

        # Create Dataset objects and dump into files
        train = Dataset('data/raw_data/sdp_data_acentors_hypernyms.train.txt', vocab_words=vocab_words,
                        vocab_poses=vocab_poses, vocab_synset=vocab_synsets, vocab_depends=vocab_depends)
        pickle.dump(train, open(constants.PICKLE_DATA + 'train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
        dev = Dataset('data/raw_data/sdp_data_acentors_hypernyms.dev.txt', vocab_words=vocab_words,
                      vocab_poses=vocab_poses, vocab_synset=vocab_synsets, vocab_depends=vocab_depends)
        pickle.dump(dev, open(constants.PICKLE_DATA + 'dev.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
        test = Dataset('data/raw_data/sdp_data_acentors_hypernyms.test.txt', vocab_words=vocab_words,
                       vocab_poses=vocab_poses, vocab_synset=vocab_synsets, vocab_depends=vocab_depends)
        pickle.dump(test, open(constants.PICKLE_DATA + 'test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        print('Load data')
        train = pickle.load(open(constants.PICKLE_DATA + 'train.pickle', 'rb'))
        dev = pickle.load(open(constants.PICKLE_DATA + 'dev.pickle', 'rb'))
        test = pickle.load(open(constants.PICKLE_DATA + 'test.pickle', 'rb'))

    # Train, Validation Split
    validation = Dataset('', process_data=False)
    train_ratio = 0.85
    n_sample = int(len(dev.words) * (2 * train_ratio - 1))
    props = ['words', 'siblings', 'positions_1', 'positions_2', 'labels', 'poses', 'synsets', 'relations', 'directions',
             'identities']
    for prop in props:
        train.__dict__[prop].extend(dev.__dict__[prop][:n_sample])
        validation.__dict__[prop] = dev.__dict__[prop][n_sample:]

    print("Train shape: ", len(train.words))
    print("Test shape: ", len(test.words))
    print("Validation shape: ", len(validation.words))

    # Get word embeddings
    embeddings = get_trimmed_w2v_vectors(constants.TRIMMED_W2V)
    model = CnnModel(
        model_name=constants.MODEL_NAMES.format('cnn', constants.JOB_IDENTITY),
        embeddings=embeddings,
        batch_size=512
    )

    # Build model
    model.build()

    model.load_data(train=train, validation=validation)
    model.run_train(epochs=constants.EPOCHS, early_stopping=constants.EARLY_STOPPING, patience=constants.PATIENCE)

    # Test on abstract
    answer = {}
    identities = test.identities
    # print(identities)
    y_pred = model.predict(test)
    for i in range(len(y_pred)):
        # print(y_pred[i])
        # print(identities[i][1])
        # print('Predict: ' + str(y_pred[i]))
        if y_pred[i] == 0:
            if identities[i][0] not in answer:
                answer[identities[i][0]] = []

            if identities[i][1] not in answer[identities[i][0]]:
                # if identities[i][1].find(str(-1)) != -1:
                #     ent1, ent2 = identities[i][1].rsplit('_', 2)
                #     if ent1 == str(-1):
                #         title_entities = [ent for ent in freq_ent[identities[i][0]] if 't' in ent]
                #         abstract_entities = [ent for ent in freq_ent[identities[i][0]] if 'a' in ent]
                #         if title_entities:
                #             for t in title_entities:
                #                 title = t[0]
                #                 other = title + '_' + ent2
                #                 answer[identities[i][0]].append(other)
                #         elif abstract_entities:
                #             for a in abstract_entities:
                #                 abstract = a[0]
                #                 other = abstract + '_' + ent2
                #                 answer[identities[i][0]].append(other)
                #         else:
                #             answer[identities[i][0]].append(identities[i][1])

                # if ent2 == str(-1):
                #     title_entities = [ent for ent in freq_ent[identities[i][0]] if 't' in ent]
                #     abstract_entities = [ent for ent in freq_ent[identities[i][0]] if 'a' in ent]
                #     if title_entities:
                #         for t in title_entities:
                #             title = t[0]
                #             other = ent1 + '_' + title
                #             answer[identities[i][0]].append(other)
                #     elif abstract_entities:
                #         for a in abstract_entities:
                #             abstract = a[0]
                #             other = ent1 + '_' + abstract
                #             answer[identities[i][0]].append(other)
                #     else:
                #         answer[identities[i][0]].append(identities[i][1])

                # else:
                answer[identities[i][0]].append(identities[i][1])

    print(
        'result: abstract: ', evaluate_bc5(answer)
    )

    result_file.write(str(evaluate_bc5(answer)))
    result_file.write('\n')
    result_file.close()


if __name__ == '__main__':
    main()

import pickle

import constants
from data_utils import *
from dataset import Dataset
from evaluate.bc5 import evaluate_bc5
from models.model_cnn import CnnModel
from sklearn.utils import shuffle


def main(times=1):
    result_file = open('data/results.py', 'w+')
    for i in range(times):
        if constants.IS_REBUILD == 1:
            print('Build data')
            # Load vocabularies
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
        props = ['words', 'positions_1', 'positions_2', 'labels', 'poses', 'synsets', 'relations', 'directions',
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
        y_pred = model.predict(test)
        for i in range(len(y_pred)):
            if y_pred[i] == 0:
                if identities[i][0] not in answer:
                    answer[identities[i][0]] = []

                if identities[i][1] not in answer[identities[i][0]]:
                    answer[identities[i][0]].append(identities[i][1])

        ev = evaluate_bc5(answer)

        print(
            'result: abstract: ', ev
        )
        result_file.write(str(ev))
        result_file.write('\n')

    result_file.close()


if __name__ == '__main__':
    main(75)

from torchtext.data.functional import generate_sp_model


def create_corpus(corpus_path, dataframe):
    corpus_set = []
    for transcript, cnt in dataframe.transcription.value_counts().iteritems():
        corpus = [transcript + '\n'] * cnt
        corpus_set.extend(corpus)

    with open(corpus_path, 'w') as f:
        f.writelines(corpus_set)


def create_vocab_model(corpus_path, vocab_size,model_prefix):
    generate_sp_model(filename=corpus_path, vocab_size=vocab_size,model_prefix=model_prefix)

import numpy as np
from matplotlib import pyplot as plt

def DTMatrix(words, TNG_train, TNG_cv, reduced_doc=20):
    selected_words = [
        "car",
        "model",
        "technology",
        "computer",
        "space",
        "language",
        "research",
        "information",
        "software",
        "data",
        "disk",
        "mac"
    ]

    selected_word_indices = [words.tolist().index(word) for word in selected_words]
    reduced_matrix = TNG_cv[0:reduced_doc, selected_word_indices].toarray()
    reduced_matrix = np.transpose(reduced_matrix)

    if np.max(TNG_cv.toarray())>1:
        plt.imshow(np.max(reduced_matrix)-reduced_matrix, cmap='gray', vmin=0, vmax=np.max(reduced_matrix)) #Count Vectorizer
    else:
        plt.imshow(1-reduced_matrix, cmap='gray', vmin=0, vmax=1)   #TFIDF

    target=[TNG_train.target_names[TNG_train.target[i]] for i in range(reduced_doc)]
    plt.xticks(range(reduced_doc), target, rotation=90)
    plt.yticks(range(len(selected_words)), selected_words)
    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)

    plt.show()


def sim_cos(cosines, TNG_train, docs_cmp=150):
    col=3
    most_similar_indices = np.argsort(cosines)[-docs_cmp-1:-1][::-1]
    most_similar_doc = [TNG_train.target_names[TNG_train.target[i]] for i in most_similar_indices]
    print ("Documento elegido: \n\t", TNG_train.target_names[TNG_train.target[0]])
    print ("Mayoes similitud coseno con otros documentos:")
    docs_cmp = int(docs_cmp/col)
    for i in range(docs_cmp):
        print("{:.5f}  {:<25}  {:.5f}  {:<25}  {:.5f}  {:<25}".format(
            cosines[most_similar_indices[i]], most_similar_doc[i],
            cosines[most_similar_indices[i+docs_cmp]], most_similar_doc[i+docs_cmp],
            cosines[most_similar_indices[i+2*docs_cmp]], most_similar_doc[i+2*docs_cmp]))

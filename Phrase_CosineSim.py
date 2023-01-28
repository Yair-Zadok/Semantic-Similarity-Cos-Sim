import gensim
import numpy as np
###################################################################################################################################################################################################################

# Loads the pre-trained Word2Vec model
model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

def semantic_similarity(phrase1, phrase2):
    # Seperates words in the phrase into digestable tokens
    tokens1 = phrase1.split()
    tokens2 = phrase2.split()


    # Calculate average embedding of the phrases
    embeddings1 = []
    for token in tokens1:
        if token in model:
            embeddings1.append(model[token])
    embeddings1 = np.array(embeddings1)
    embeddings1 = embeddings1.mean(axis=0) # Calculates the mean of all the embeddings to create a vector of the whole phrase


    embeddings2 = []
    for token in tokens2:
        if token in model:
            embeddings2.append(model[token])
    embeddings2 = np.array(embeddings2)
    embeddings2 = embeddings2.mean(axis=0) 
    

    # Calculating the cosine similarity
    dot_product = np.dot(embeddings1, embeddings2)
    magnitude1 = np.linalg.norm(embeddings1) # Calculates the magnitude of the vector
    magnitude2 = np.linalg.norm(embeddings2)
    cos_similarity = dot_product / (magnitude1 * magnitude2) # Computes the cosine of the angle between the two angles
    
    return cos_similarity


###################################################################################################################################################################################################################

# Test examples
phrase1 = "I like ice cream"
phrase2 = "I dislike ice cream"

print("The semantic similarity is:", semantic_similarity("geography", "country"))

###################################################################################################################################################################################################################


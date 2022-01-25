# Lecture 2: Object Representations

* Data science involves using scientific tools, algorithms and data models to statistically analyse and extract knowledge from large volumes of processed/unprocessed collected data.

* Mathematically, an object can be represented in 2 ways:
  * Object Representation: To depict isolated objects like vectors (words in a document), sequences (DNA), etc.
  * Network Represenation: To show interaction among objects like graphs (friendship network, road connections), etc.

* The most common representation for documents is called 'bag of words' representation.
  * First the document sentences are tokenized.
  * Then their base forms are derived via stemming and lemmatization.
  * The collected lexemes are then converted to n-gram or skip gram model. As n increases, the size of word dictionary and thus, the memory required to store it increases but contextual information is captured more efficiently.

``` {note}
It implies that as we move from unigram to bigram, the connection between mathematical similarity and intuitional similarity becomes tighter.

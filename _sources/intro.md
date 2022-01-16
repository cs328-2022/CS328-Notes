## Course Description
CS328 aims to introduce students to the various statistical and algorithmic ideas that underlie the field of data science.  At the end of the course,  students will be familiar with algorithms to extract patterns from medium to large data sets. They should also be comfortable in the various model statistical model fitting techniques as well as understand how to argue about statistical significance. Students will also get exposed to practical tools (e.g. numpy/scipy/matplotlib/sklearn if using python, or analogous tools in other languages). This course will be valuable to anyone interested in continuing studies/working in data analytics or machine learning.

**Instructor**: Anirban Dasgupta, **Office**: AB 6/407c. Please email for appointment.

**Teaching Assistant**: Shrutimoy Das, Sachin Yadav


## Lecture Schedule
There are roughly 40 lecture hours in the calendar. The following is a tentative order in which the topics will be covered.

````{panels}
Foundations
^^^
   - Data representation, distance measures
   - Central limit theorem
   - Random variables and tail inequalities, hashing, balls and bins.
   - Practical example of hashing-- MinHash
---

Clustering and low-rank approximations
^^^
   - k-means, k-center, Lloyds algorithm, k-means++
   - Clustering in graphs -- expansion, conductance, modularity.
   - Spectral algorithms for expansion and conductance.
   - Louvain algorithm for modularity.
   - Learning mixture models -- Gaussians.
   - SVD and its applications, other matrix factorizations.
---

Dealing with massive data
^^^
   - Efficient data summaries -- Bloom filters, bit arrays.
   - Streaming model: samples and sketches -- reservoir sampling, counting distinct elements, heavy hitter data structure(Misra-Gries, Count-Min, Count-Sketch)
---

Random Walks
^^^
   - Random walks and convergence, connection with eigenvalues
   - PageRank, HITS and their interpretations
   - Gibbs sampling and Markov chains
---

Drawing inference from the data
^^^
   - Sampling, estimation, confidence intervals, bootstrapping
   - Hypothesis testing and its variants-- multiple hypothesis testing, Bayes Factor
   - Linear regression and its generalizations, model evaluation, goodness of fit tests
````

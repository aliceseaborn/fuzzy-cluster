# FuzzyCluster

**Brief:**
This repository represents my work on an algorithm to quickly cluster arrays of records by their similarity. It offers a high degree of configurability by the end user with respect to what array elements matter more for the comparison. This system is still new and it under development but this publication marks the beginning of the optimization process to bring the time complexity of the algorithm back down to Earth. I am excited to see what can be made of this project.

**Background:**
A mid-sized insurance company based in Peoria IL was in need of a technique to group together the policies stored in their database by the unique client who owned the policy. Unfortunately, insurance laws required that new policies be entered manually into the database. Consequently, many of their policies had misspelled names, inverted characters in their SSNs, improper addresses, etc. In order to counter this challenge, they were in need of a way to fuzzy-cluster their policies by the client to account for the misspellings.

**Technique:**
Using a pseudo-K-Means clustering technique involving text-based centroids, policies were scored by their Levenshtein distances to each client cluster. The policy was added to the cluster if it was sufficiently similar to the cluster's centroid based on the scoring parameters set by the user/scientist. The account became the centroid of its own group if no existing cluster was sufficiently similar. Each iteration of the cycle, each cluster is reevaluated. During the evaluation, the policy with the greatest similarity to the other policies in its group became the centroid. Ergo, the centroid of each group was the best representative of the data within the group.

**Outcome:**
As a result of this algorithm, a list of thousands of clients could be accurately clustered according to custom similarity parameters. However, this algorithm has an exponential time-complexity and could not quickly cluster lists of hundreds of thousands of clients without superior hardware to what we had available.

import numpy as np   # bring in NumPy and give it the short name np
from sklearn.preprocessing import StandardScaler   # bring in the scaler: it puts every clue on the same ruler
from sklearn.cluster import KMeans, AgglomerativeClustering   # bring in two clustering tools: k-means and hierarchical clustering
from sklearn.metrics import silhouette_score   # bring in the silhouette score: it grades how clean the groups are
from sklearn.decomposition import PCA   # bring in PCA: it squeezes many clues into fewer combined clues

# THE DATA — 12 students, two clues each: hours studied and average quiz score. There are NO labels this time.
students = np.array([[1, 42], [2, 48], [2, 55], [3, 52], [4, 63], [5, 68],
                     [5, 72], [6, 70], [7, 84], [8, 88], [8, 93], [9, 95]])   # clues only: nobody tells the program the right groups

# SCALING — quiz scores run to 95 but hours stop at 9, so without scaling the scores would shout and the hours would whisper.
scaler = StandardScaler()   # create the scaler, still unused
scaled = scaler.fit_transform(students)   # measure both clues and re-express them on one shared ruler

# CHOOSING HOW MANY GROUPS — try 1 to 5 groups and watch the leftover scatter, called inertia, shrink.
for k in range(1, 6):   # try each number of groups from 1 up to 5
    trial = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)   # cluster the students into k groups and keep the result
    print("Inertia with", k, "groups:", round(trial.inertia_, 1))   # the leftover scatter: lower means tighter groups

# K-MEANS — settle on 3 groups, the bend in the elbow, and let the algorithm sort the students.
groups_model = KMeans(n_clusters=3, random_state=42, n_init=10)   # create the k-means model, still untrained
groups = groups_model.fit_predict(scaled)   # cluster: each student is handed to the group whose centre is nearest
print("Group for each student:", groups)   # the group number k-means gave each of the 12 students
print("Silhouette score:", round(silhouette_score(scaled, groups), 2))   # how clean the groups are: near 1 is tidy, near 0 is messy
centres = scaler.inverse_transform(groups_model.cluster_centers_)   # turn the group centres back into real hours and real quiz scores
print("Group centres (hours, quiz score):", np.round(centres, 1))   # the middle student of each group, in real units

# A SECOND OPINION — hierarchical clustering builds the groups by merging the closest students step by step.
h_groups = AgglomerativeClustering(n_clusters=3).fit_predict(scaled)   # cluster the same students the merging way
print("Hierarchical group for each student:", h_groups)   # the group number the merging method gave each student
same = sum(1 for i in range(12) for j in range(i + 1, 12) if (groups[i] == groups[j]) == (h_groups[i] == h_groups[j]))   # count student pairs both methods treat the same way
print("Student pairs the two methods agree on:", same, "of 66")   # out of all 66 pairs of students

# PCA — squeeze the two clues into ONE combined score and check how much of the pattern survives.
pca = PCA(n_components=1)   # create the PCA tool, set to build a single combined clue
combined = pca.fit_transform(scaled)   # squeeze every student's two clues into one number
print("Variance kept by the combined score:", round(pca.explained_variance_ratio_[0], 2))   # the share of the pattern the one number keeps
print("Combined score for each student:", np.round(combined.ravel(), 1))   # the squeezed scores: low for strugglers, high for stars

# ONE NEW STUDENT — 2 hours of study but a quiz score of 95: which group, and does the student fit it?
new_student = scaler.transform([[2, 95]])   # put the new student on the same ruler as everyone else
print("New student's group:", groups_model.predict(new_student))   # the group whose centre is nearest to the new student
own_distances = np.linalg.norm(scaled - groups_model.cluster_centers_[groups], axis=1)   # how far each known student sits from their own group's centre
new_distance = np.linalg.norm(new_student - groups_model.cluster_centers_[groups_model.predict(new_student)[0]])   # how far the new student sits from the nearest centre
print("New student's distance to the nearest centre:", round(new_distance, 2))   # compare this with the known students below
print("Farthest known student from their centre:", round(own_distances.max(), 2))   # the most distant of the 12 known students
print("Verdict:", "typical" if new_distance <= own_distances.max() else "UNUSUAL — an anomaly, far from every group")   # beyond the farthest known student means odd one out

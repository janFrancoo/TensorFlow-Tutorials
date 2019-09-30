import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define parameters
num_points = 200
num_clusters = 3
num_iteration = 100

# Create points and pick centroids randomly
points = tf.constant(np.random.uniform(0, 10, (num_points, 2)))
centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [num_clusters, -1]))

# Expand points and centroids into 3 dimensions
points_expanded = tf.expand_dims(points, 0)
centroids_expanded = tf.expand_dims(centroids, 1)

# Define a distance metric and assign clusters
distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
assignments = tf.argmin(distances, 0)

# Update centroids
means = []
for c in range(num_clusters):
    means.append(tf.reduce_mean(tf.gather(points, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])),
                                reduction_indices=[1]))

new_centroids = tf.concat(means, 0)
update_centroids = tf.assign(centroids, new_centroids)

# Define init func
init = tf.global_variables_initializer()

# Start session
with tf.Session() as sess:
    sess.run(init)
    for step in range(num_iteration):
        [_, centroid_values, points_values, assignment_values] = sess.run(
            [update_centroids, centroids, points, assignments])

    print("centroids", centroid_values)

# Visualize results
plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
plt.show()

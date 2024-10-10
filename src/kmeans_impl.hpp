#include <iostream>
#include <vector>
#include "rng.h"
#include <utility>

struct PointView {
    const double* point;
    size_t len;
};

struct DataSet {
    const std::vector<double> &data;
    size_t rows;
    size_t cols;

    PointView get_point(size_t rowIdx) const {
        return {
            &data[rowIdx * rows],
            cols
        };
    }
};

size_t run_kmeans(Rng &rng, DataSet dataSet, size_t amtCentroids) {
    std::vector<size_t> cluster_map = std::vector<size_t>(dataSet.rows, -1);
    auto centroid_indices = std::vector<size_t>(amtCentroids);
    rng.pickRandomIndices(dataSet.rows, centroid_indices);
    
    auto centroids = std::vector<PointView>(amtCentroids);
    for (const auto c_idx : centroid_indices)
        centroids.push_back(dataSet.get_point(c_idx));

    bool changed = true;
    while (changed) {
        changed = false;
        for (int pointIdx = 0; pointIdx < dataSet.rows; pointIdx++) {
            auto row = dataSet.get_point(pointIdx);
            const auto [newCluster, dist] = closest_centroid_and_dist(centroids, row);

            if (newCluster != cluster_map[pointIdx]) {
                cluster_map[pointIdx] = newCluster;
                changed = true;
            }
        }
        if (changed) {
            for (int i = 0; i < amtCentroids; i++) {
                // reculaculate centroid position
            }
        } 
    }

    return 0;
}

std::pair<size_t, double> closest_centroid_and_dist(const std::vector<PointView>& centroids, const PointView row) {
    double best_dist =  std::numeric_limits<double>::max(); // can only get better
    size_t best_index = 0;

    for (size_t c_idx = 0; c_idx< centroids.size(); c_idx++){
        double  dist_sum_sqrd= 0;
        const auto centroid = centroids[c_idx];
        for (size_t i; i < row.len; i++){
            dist_sum_sqrd += centroid.point[i] * row.point[i];
        }
        if (dist_sum_sqrd < best_dist){
            best_dist = dist_sum_sqrd;
            best_index = 0; // index of centroid
        }
    }

    return {best_index, best_dist};
}
#include <iostream>
#include <vector>
#include "rng.h"
#include <utility>
#include <algorithm>

struct PointView {
    const double* point;
    size_t len;
    size_t idx;
};

struct DataSet {
    const std::vector<double> &data;
    size_t rows;
    size_t cols;

    PointView get_point(size_t rowIdx) const {
        return {
            &data[rowIdx * rows],
            cols,
            rowIdx,
        };
    }
};

size_t run_kmeans(Rng &rng, const DataSet& data_set, size_t amt_centroids) {
    std::vector<size_t> cluster_map = std::vector<size_t>(data_set.rows, -1);

    auto centroid_indices = std::vector<size_t>(amt_centroids);
    rng.pickRandomIndices(data_set.rows, centroid_indices);
    
    auto centroids = std::vector<PointView>(amt_centroids);
    for (const auto c_idx : centroid_indices)
        centroids.push_back(data_set.get_point(c_idx));

    bool changed = true;
    while (changed) {
        changed = false;
        
        for (int pointIdx = 0; pointIdx < data_set.rows; pointIdx++) {
            auto row = data_set.get_point(pointIdx);
            const auto [newCluster, dist] = closest_centroid_and_dist(centroids, row);

            if (newCluster != cluster_map[pointIdx]) {
                cluster_map[pointIdx] = newCluster;
                changed = true;
            }
        }
        if (changed) {
            for (int i = 0; i < amt_centroids; i++) {
                // reculaculate centroid position
                auto point = average_of_points_with_cluster(centroids[i], cluster_map, data_set);
                //centroids[i] = ;
            }
        } 
    }

    return 0;
}

std::pair<size_t, double> closest_centroid_and_dist(const std::vector<PointView>& centroids, const PointView row) {
    double best_dist =  std::numeric_limits<double>::max(); // can only get better
    size_t best_index = 0;

    for (size_t c_idx = 0; c_idx< centroids.size(); c_idx++){
        double  dist_sum_sqrd = 0;
        const auto centroid = centroids[c_idx];

        // euclidic distance: sqrt((x1 - y1)² + (x2 - y2)²)
        // where x and y are point in R²
        for (size_t i; i < row.len; i++){
            dist_sum_sqrd += centroid.point[i] - row.point[i];
        }

        if (dist_sum_sqrd < best_dist){
            best_index = c_idx; 
            // index of centroi d
            best_dist = dist_sum_sqrd;
        }
    }

    return { best_index, best_dist };
}

// ik heb geen idee of dit klopt in godsnaam
std::vector<double> average_of_points_with_cluster(const PointView& centroid, std::vector<size_t> cluster_map, const DataSet& data_set) {
    int count = 0;
    auto avg_point = std::vector<double>(centroid.len);
    for (int i = 0; i < cluster_map.size(); i++) {
        size_t cluster = cluster_map[i];
        PointView p = data_set.get_point(i);
        if (cluster == centroid.idx) {
            count++;
            std::transform (avg_point.begin(), avg_point.end(), &p.point, avg_point.begin(), std::plus<int>());
        }
    }
    std::transform(avg_point.begin(), avg_point.end(), avg_point.begin(), [count](int x) { return x / count; });
    return avg_point;
}
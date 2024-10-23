#include <cstddef>
#include <iomanip>
#include <ostream>
#include <vector>
#include "rng.h"
#include <utility>
#include <algorithm>
#include <functional>
#include <iostream>

struct PointView {
    const double* point;
    size_t len;
};

std::ostream& operator<<(std::ostream& o, const PointView &v){
    o << std::setprecision(15);
    for (size_t i = 0; i < v.len-1; i++){
        o << v.point[i] << ", ";
    }
    o << v.point[v.len-1];
    return o;
}

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T> &v){
    o << std::setprecision(15);
    for (size_t i = 0; i < v.size()-1; i++)
        o << v[i] << ", ";
    o << v[v.size()-1];
    return o;
}

struct DataSet {
    const std::vector<double> &data;
    const size_t rows;
    const size_t cols;

    // 2D array in 1D vector how to get a view over a row???
    // say the vector of 2D points looks like this
    // rows = 4, cols = 2
    // [(10, 10), (20, 20), (30, 30), (40, 40)]
    // or: [10, 10, 20, 20, 30, 30, 40, 40]
    // to get row 2 (zero-indexed 1) (or (20, 20)) we do:
    // we get address of the first 20: 1 * 2 or in other terms
    // cols * rowIdx
    PointView get_point_view(size_t rowIdx) const {
        return {
            &data[rowIdx * cols],
            cols,
        };
    }

    // returns a copy
    std::vector<double> get_point(size_t row_idx) const {
        auto pointBegin = data.begin() + (row_idx * cols);
        auto pointEnd = pointBegin + cols;
        return std::vector<double>(pointBegin, pointEnd);
    }
};


// for a point find the closest centroid and return that distance with the index of that centroid
std::pair<size_t, double> closest_centroid_and_dist(const double* centroids, const PointView point, size_t amtCentroids) {
    double bestDist =  std::numeric_limits<double>::max(); // can only get better
    size_t bestIndex = -1;

    for (size_t cIdx = 0; cIdx < amtCentroids; cIdx++){
        double  distSumSqrd = 0;
        // const auto& centroid = centroids[cIdx];

        // euclidic distance: sqrt((x1 - y1)² + (x2 - y2)²)
        // where x and y are points in R²
        for (size_t i = 0; i < point.len; i++){
            auto pointDiff = centroids[cIdx * point.len + i] - point.point[i];
            distSumSqrd += pointDiff * pointDiff;
        }

        if (distSumSqrd < bestDist) {
            bestIndex = cIdx;
            // index of centroi d
            bestDist = distSumSqrd;
        }
    }
    return { bestIndex, bestDist };
}


std::vector<double> average_of_points_with_cluster(const size_t centroidIdx, const size_t* clusterMap, const DataSet& dataSet) {
    size_t count = 0;
    auto avgPoint = std::vector<double>(dataSet.cols, 0);
    for (int i = 0; i < dataSet.rows; i++) {
        size_t clusterIdx = clusterMap[i];
        PointView v = dataSet.get_point_view(i);
        if (clusterIdx == centroidIdx) {
            count++;
            std::transform(
                avgPoint.cbegin(), avgPoint.cend(), // iterate over avgPoint and transform in place
                v.point, avgPoint.begin(), // iterate over 2 iterators and perform binary_op
                std::plus<double>{} // binary_op
            );
        }
    }
    std::transform(avgPoint.cbegin(), avgPoint.cend(), avgPoint.begin(), [count](auto x) { return x / count; });
    return avgPoint;
}


struct KMeansResult{
    size_t steps;
    double bestDistSumSqrd;
    std::vector<size_t> bestCentroidsIndices;
};

KMeansResult run_kmeans(
    Rng &rng,
    const DataSet& dataSet,
    const size_t amtCentroids,
    FileCSVWriter& clustersDebugFile,
    FileCSVWriter& centroidDebugFile
) {
    double bestdistSqrdSum = std::numeric_limits<double>::max(); 
    size_t bestCentroidsIndices[dataSet.rows] = { 0 };
    // std::vector<size_t> bestCentroidsIndices{};
    // cluster map maps points to their cluster
    // the value is the index of the cluster
    // the index of that value is the point
    // std::vector<size_t> centroidMap = std::vector<size_t>(dataSet.rows, -1);
    size_t centroidMap[dataSet.rows] = { 0 };

    // first centroid points are
    auto centroidsIndices = std::vector<size_t>(amtCentroids);
    rng.pickRandomIndices(dataSet.rows, centroidsIndices);

    double centroids[amtCentroids][dataSet.cols] = { { 0 } };
    // auto centroids = std::vector<Centroid>{};
    int c = 0;
    for (const auto cIdx : centroidsIndices) {
        auto centroid = dataSet.get_point(cIdx);
        std::copy(centroid.begin(), centroid.end(), centroids[c]);
        c++;
    }

    bool changed = true;
    size_t steps = 0;

    while (changed) {
        if (centroidDebugFile.is_open()) {
            for (const double* centroid: centroids) {
                auto v = std::vector<double>(centroid, centroid + dataSet.cols);
                centroidDebugFile.write(v, dataSet.cols);
            }
        }
        if (clustersDebugFile.is_open()) {
            auto v = std::vector<size_t>(centroidMap, centroidMap + dataSet.rows);
            clustersDebugFile.write(v);
        }

        changed = false;
        double distSqrdSum = 0;
        steps++;

        for (int pointIdx = 0; pointIdx < dataSet.rows; pointIdx++) {
            auto row = dataSet.get_point_view(pointIdx);
            const auto result = closest_centroid_and_dist(&centroids[0][0], row, amtCentroids);
            const auto newCluster = result.first;
            distSqrdSum += result.second;
            if (newCluster != centroidMap[pointIdx]) {
                centroidMap[pointIdx] = newCluster;
                changed = true;
            }
        }
        if (changed) {
            for (int i = 0; i < amtCentroids; i++) {
                // reculaculate centroid position
                auto n = average_of_points_with_cluster(i, centroidMap, dataSet);
                memcpy(centroids[i], &n[0], dataSet.cols * sizeof(size_t));
                // std::cout << n << std::endl;
            }
        }

        if (distSqrdSum < bestdistSqrdSum){
            memcpy(bestCentroidsIndices, centroidMap, dataSet.rows * sizeof(size_t));
            bestdistSqrdSum = distSqrdSum;
        }
    }

    return {
        steps,
        bestdistSqrdSum,
        std::vector<size_t>(bestCentroidsIndices, bestCentroidsIndices + dataSet.rows)
    };
}

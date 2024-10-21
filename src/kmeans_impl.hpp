#include <cstddef>
#include <iomanip>
#include <ostream>
#include <vector>
#include "rng.h"
#include <utility>
#include <algorithm>
#include <functional>
#include <iostream>
#include "CSVWriter.hpp"

struct PointView {
    const double* point;
    size_t len;
};

// centroid is just an N-dim point
using Centroid = std::vector<double>;

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

struct CentroidPointMapping{
    
    // cluster map maps points to their cluster
    // the value is the index of the cluster
    // the index of that value is the point
    std::vector<size_t> centroidMap;
    // cluster index to its points 
    // cluster index -> collection of points
    std::vector<std::vector<PointView>> centroidToPointMap;
    /*
        
    */

    CentroidPointMapping(size_t amountPoints, size_t amtCentriods): centroidMap(std::vector<size_t>(amountPoints, 0)), centroidToPointMap(amtCentriods) {
        for (auto& points : centroidToPointMap){
            points.reserve((size_t) amountPoints / amtCentriods); // pre allocate
        }
    }

    void clear(){
        centroidMap.clear();
        centroidToPointMap.clear();
    }

    size_t centroid_map_size() const{
        return centroidMap.size();
    }


    const std::vector<PointView>& points_in_cluster(size_t clusterIdx) const{
        return centroidToPointMap[clusterIdx];
    }
    void remap_point_to_centroid(size_t pointIdx, PointView pv, size_t newCentroidIdx, size_t oldCentroidIdx){
        /// removed point
        auto& oldCluster = centroidToPointMap[oldCentroidIdx];
        auto removed = std::remove_if(oldCluster.begin(), oldCluster.end() , [pv](PointView to_remove){return to_remove.point == pv.point;});
        oldCluster.erase(removed, oldCluster.end());
        centroidToPointMap[newCentroidIdx].push_back(pv);
        centroidMap[pointIdx] = newCentroidIdx;
    }

    size_t centroid_of_point(size_t pointIdx) const{
        return centroidMap[pointIdx];
    }
};

struct DataSet {
    const std::vector<double> &data;
    size_t rows;
    size_t cols;

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

struct KMeansResult{
    size_t steps;
    double bestDistSumSqrd;
    std::vector<size_t> bestCentroidsIndices;
};

class KMeans{
    DataSet dataSet;
    size_t amtCentroids;
    CentroidPointMapping mapping;
    std::vector<Centroid> centroids;
    std::vector<size_t> centroidsIndices;    
    
    public:


    KMeans(DataSet ds, size_t amountCentroids)
        : dataSet(ds)
        , amtCentroids(amountCentroids)
        , mapping(CentroidPointMapping(ds.rows, amtCentroids))
        , centroidsIndices(std::vector<size_t>(amtCentroids))
        , centroids(std::vector<Centroid>(amtCentroids)){
            std::cout << "CENTROID INDICES SIZE: " << centroidsIndices.size() << std::endl;
            std::cout << "CENTROIDS SIZE: " << centroids.size() << std::endl;
    }

    void clear_vectors(){
        mapping.clear();
        centroids.clear();
        centroidsIndices.clear();
    }

    // for a point find the closest centroid and return that distance with the index of that centroid
    std::pair<size_t, double> closest_centroid_and_dist(const PointView point) const {
        double bestDist =  std::numeric_limits<double>::max(); // can only get better
        size_t bestIndex = 0;

        for (size_t cIdx = 0; cIdx < centroids.size(); cIdx++){
            double  distSumSqrd = 0;
            const auto centroid = centroids[cIdx];

            // euclidic distance: sqrt((x1 - y1)² + (x2 - y2)²)
            // where x and y are points in R²
            for (size_t i = 0; i < point.len; i++){
                auto pointDiff = centroid[i] - point.point[i];
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


    std::vector<double> average_of_points_with_cluster(const size_t centroidIdx) const {
        size_t count = 0;
        auto avgPoint = std::vector<double>(dataSet.cols, 0);
        for (const auto pointView : mapping.points_in_cluster(centroidIdx)){
            count++;
            std::transform(
                avgPoint.cbegin(), avgPoint.cend(), // iterate over avgPoint and transform in place
                pointView.point, avgPoint.begin(), // iterate over 2 iterators and perform binary_op
                std::plus<double>{} // binary_op
            );
        }
        if (count){
            std::transform(avgPoint.cbegin(), avgPoint.cend(), avgPoint.begin(), [count](auto x) { return x / count; });
        }
        return avgPoint;
    }




    KMeansResult run_kmeans(
        Rng &rng,
        FileCSVWriter& clustersDebugFile,
        FileCSVWriter& centroidDebugFile
    ) {
        // clear but keep allocations
        clear_vectors();
        double bestdistSqrdSum = std::numeric_limits<double>::max(); // can only get better

    
        rng.pickRandomIndices(dataSet.rows, centroidsIndices);
        for (const auto cIdx : centroidsIndices) {
            auto centroid = dataSet.get_point(cIdx);
            centroids[cIdx] = centroid;
        }

        std::vector<size_t> bestCentroidsIndices = mapping.centroidMap;

        bool changed = true;
        size_t steps = 0;

        while (changed) {
            if (centroidDebugFile.is_open()) {
                for (const auto& centroid: centroids) {
                    centroidDebugFile.write(centroid, dataSet.cols);
                }
            }
            if (clustersDebugFile.is_open()) {
                clustersDebugFile.write(mapping.centroidMap);
            }

            changed = false;
            double distSqrdSum = 0;
            steps++;

            for (int pointIdx = 0; pointIdx < dataSet.rows; pointIdx++) {
                auto pointView = dataSet.get_point_view(pointIdx);
                const auto result = closest_centroid_and_dist(pointView);
                distSqrdSum += result.second;

                const auto newCluster = result.first; 
                const auto oldCluster = mapping.centroid_of_point(pointIdx);    
                if (newCluster != oldCluster) {
                    mapping.remap_point_to_centroid(pointIdx, pointView, newCluster, oldCluster);
                    changed = true;
                }
            }
            if (changed) {
                for (int i = 0; i < amtCentroids; i++) {
                    // reculaculate centroid position
                    centroids[i] = std::move(average_of_points_with_cluster(i));
                }
            }

            if (distSqrdSum < bestdistSqrdSum){
                bestCentroidsIndices = mapping.centroidMap;
                bestdistSqrdSum = distSqrdSum;
            }
        }

        return {
            steps,
            bestdistSqrdSum,
            bestCentroidsIndices
        };
    }
};

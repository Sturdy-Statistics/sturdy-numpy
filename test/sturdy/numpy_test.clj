(ns sturdy.numpy-test
  (:require [clojure.test :refer [deftest is testing]]
            [tech.v3.dataset :as ds]
            [sturdy.numpy :as sn]
            [sturdy.numpy.test-utils :refer [resource-path]]))

(set! *warn-on-reflection* true)

(deftest public-api-wrappers-test
  (let [path (resource-path "shape_2x3__dtype_i4.npy")]
    (testing "npy->vec"
      (is (= [[-5 12 29] [46 63 80]] (sn/npy->vec path))))

    (testing "npy->dataset"
      (let [ds (sn/npy->dataset path)]
        (is (= 2 (ds/row-count ds)))
        (is (= [:c1 :c2 :c3] (ds/column-names ds)))))

    (testing "npy->primitive"
      (let [{:keys [shape dtype fortran? data]} (sn/npy->primitive path)]
        (is (= [2 3] shape))
        (is (= :i4 dtype))
        (is (= false fortran?))
        (is (some? data))))

    (testing "npy->dataset-rowlists"
      (let [ds (sn/npy->dataset-rowlists path)]
        (is (= 2 (ds/row-count ds)))
        (is (= [:c1] (ds/column-names ds)))))

    (testing "npy->dataset-unnested-nz"
      (let [ds (sn/npy->dataset-unnested-nz path)]
        (is (= 6 (ds/row-count ds)))
        (is (= #{:row_no :col_no :val} (set (ds/column-names ds))))))))

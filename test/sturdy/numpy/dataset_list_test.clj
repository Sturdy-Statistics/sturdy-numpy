(ns sturdy.numpy.dataset-list-test
  (:require [clojure.test :refer [deftest is testing]]
            [tech.v3.dataset :as ds]
            [tech.v3.datatype :as dtype]
            [sturdy.numpy.dataset-list :refer [npy->dataset-rowlists]]
            [sturdy.numpy.test-utils :refer [resource-path]]))

(set! *warn-on-reflection* true)

(deftest npy->dataset-rowlists-test
  (testing "Successfully reads 2D C-order arrays"
    (let [ds (npy->dataset-rowlists (resource-path "manual_2x3__dtype_i4.npy"))
          c1 (get ds :c1)]
      (is (= 2 (ds/row-count ds)))
      (is (= [:c1] (ds/column-names ds)))
      (is (= :object (dtype/elemwise-datatype c1)))
      (let [row0 (c1 0)
            row1 (c1 1)]
        (is (= [1 2 3] (vec row0)))
        (is (= [4 5 6] (vec row1))))))

  (testing "Successfully wraps unsigned 2D arrays"
    (let [ds (npy->dataset-rowlists (resource-path "shape_2x3__dtype_u4.npy"))
          c1 (get ds :c1)
          row0 (c1 0)]
      (is (= 2 (ds/row-count ds)))
      (is (= :uint32 (dtype/elemwise-datatype row0)))))

  (testing "Rejects 1D arrays"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"requires a 2D array"
                          (npy->dataset-rowlists (resource-path "shape_10___dtype_u4.npy")))))

  (testing "Rejects Fortran-order arrays"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"does not currently support Fortran-order"
                          (npy->dataset-rowlists (resource-path "shape_2x3__dtype_u4__order_F.npy"))))))

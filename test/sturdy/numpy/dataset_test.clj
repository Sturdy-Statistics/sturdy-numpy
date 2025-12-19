(ns sturdy.numpy.dataset-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [clojure.string :as str]
   [tech.v3.dataset :as ds]
   [tech.v3.datatype :as dtype]
   [sturdy.numpy.dataset :refer [npy->dataset]]
   [sturdy.numpy.test-utils :refer [resource-path]]
   [sturdy.numpy.read-test :refer [expected-vals-1d]]))

(defn expected-cols
  "Return a vector of expected column vectors for given shape+dtype,
   assuming the file data is row-major (C-order) and we build a columnar dataset."
  [shape dtype]
  (let [n    (reduce * 1 shape)
        flat (expected-vals-1d dtype n)]
    (if (= 1 (count shape))
      ;; treat 1D as a single column
      [(vec flat)]
      (let [[rows cols] shape]
        (mapv (fn [j]
                (mapv (fn [r] (nth flat (+ (* r cols) j)))
                      (range rows)))
              (range cols))))))

(defn colname [j]
  (keyword (str "c" (inc j))))

(defn ds-col->vec
  "Extract a dataset column as a persistent vector (works for primitive columns)."
  [ds cname]
  ;; ds provides map-like access: (get ds :c1) -> column
  (vec (get ds cname)))

;; --- Tests ---

(deftest npy->dataset-dtypes-on-2x3
  (testing "Builds a dataset with correct columns for each dtype (shape 2x3)"
    (doseq [dtype [:u1 :u2 :u4 :i1 :i2 :i4 :i8 :f4 :f8]]
      (let [fname (format "shape_2x3__dtype_%s.npy" (name dtype))
            ds    (npy->dataset (resource-path fname))
            exp   (expected-cols [2 3] dtype)]
        (is (= 2 (ds/row-count ds)) (str "row count mismatch for " fname))
        (is (= 3 (count (ds/column-names ds))) (str "col count mismatch for " fname))
        (doseq [j (range 3)]
          (is (= (exp j) (ds-col->vec ds (colname j)))
              (str "col mismatch for " fname " col " (inc j))))))))

(deftest npy->dataset-shapes-on-u4
  (testing "Builds a dataset with correct shape/columns for dtype u4"
    (doseq [shape [[1] [10] [1 10] [2 3] [10 1] [10 12]]]
      (let [shape-tag (if (= 1 (count shape))
                        (format "%d_" (first shape))
                        (str/join "x" shape))
            fname (format "shape_%s__dtype_u4.npy" shape-tag)
            ds    (npy->dataset (resource-path fname))
            exp   (expected-cols shape :u4)]
        (is (= (first shape) (ds/row-count ds)) (str "row count mismatch for " fname))
        (is (= (count exp) (count (ds/column-names ds))) (str "col count mismatch for " fname))
        (doseq [j (range (count exp))]
          (is (= (exp j) (ds-col->vec ds (colname j)))
              (str "col mismatch for " fname " col " (inc j))))))))

(deftest npy->dataset-manual-small
  (testing "Manual fixture for eyeballing (2x3 i4)"
    (let [ds (npy->dataset (resource-path "manual_2x3__dtype_i4.npy"))]
      (is (= 2 (ds/row-count ds)))
      (is (= [1 4] (ds-col->vec ds :c1)))
      (is (= [2 5] (ds-col->vec ds :c2)))
      (is (= [3 6] (ds-col->vec ds :c3))))))

(deftest npy->dataset-fortran-shapes-on-u4
  (testing "Builds a dataset correctly for Fortran-order u4 fixtures"
    (doseq [[shape fname] [[[2 3]   "shape_2x3__dtype_u4__order_F.npy"]
                           [[10 12] "shape_10x12__dtype_u4__order_F.npy"]]]
      (let [ds  (npy->dataset (resource-path fname))
            exp (expected-cols shape :u4)]
        (is (= (first shape) (ds/row-count ds)) (str "row count mismatch for " fname))
        (is (= (count exp) (count (ds/column-names ds))) (str "col count mismatch for " fname))
        (doseq [j (range (count exp))]
          (is (= (exp j) (ds-col->vec ds (colname j)))
              (str "col mismatch for " fname " col " (inc j))))))))

(deftest npy->dataset-big-endian-on-2x3
  (testing "Builds dataset correctly for big-endian fixtures (shape 2x3)"
    (doseq [[dtype fname]
            [[:i2 "shape_2x3__dtype_i2__endian_B.npy"]
             [:u4 "shape_2x3__dtype_u4__endian_B.npy"]
             [:f8 "shape_2x3__dtype_f8__endian_B.npy"]]]
      (let [ds  (npy->dataset (resource-path fname))
            exp (expected-cols [2 3] dtype)]
        (is (= 2 (ds/row-count ds)) (str "row count mismatch for " fname))
        (is (= 3 (count (ds/column-names ds))) (str "col count mismatch for " fname))
        (doseq [j (range 3)]
          (is (= (exp j) (ds-col->vec ds (colname j)))
              (str "col mismatch for " fname " col " (inc j))))))))

(deftest npy->dataset-preserves-unsigned-dtypes
  (testing "Unsigned numpy dtypes are represented as uint* columns"
    (doseq [[tp expected]
            [[:u1 :uint8]
             [:u2 :uint16]
             [:u4 :uint32]]]
      (let [fname (format "shape_2x3__dtype_%s.npy" (name tp))
            dataset (npy->dataset (resource-path fname))
            col (first (ds/columns dataset))]
        (is (= expected (dtype/elemwise-datatype col))
            (str "dtype mismatch for " fname))))))

(deftest npy->dataset-preserves-column-dtypes
  (testing "npy->dataset produces expected tech.datatype column dtypes"
    (doseq [[tp expected]
            [[:u1 :uint8]
             [:u2 :uint16]
             [:u4 :uint32]
             [:i1 :int8]
             [:i2 :int16]
             [:i4 :int32]
             [:i8 :int64]
             [:f4 :float32]
             [:f8 :float64]]]
      (let [fname (format "shape_2x3__dtype_%s.npy" (name tp))
            dataset (npy->dataset (resource-path fname))
            col (first (ds/columns dataset))]
        (is (= expected (dtype/elemwise-datatype col))
            (str "dtype mismatch for " fname " expected=" expected))))))

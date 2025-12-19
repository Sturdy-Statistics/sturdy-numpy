(ns sturdy.numpy.header-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [sturdy.fs :as sfs]
   [sturdy.numpy.test-utils :refer [resource-path]]
   [sturdy.numpy.header :refer [parse-npy-header]]))

(deftest parse-npy-header-basic-1d
  (testing "parse-npy-header parses a 1D u4 fixture"
    (let [bs  (sfs/slurp-bytes (resource-path "shape_10___dtype_u4.npy"))
          hdr (parse-npy-header bs)]
      (is (= "<u4" (:descr hdr)))
      (is (= :little (:byte-order hdr)))
      (is (= "u" (:kind hdr)))
      (is (= 4 (:size hdr)))
      (is (= :u4 (:dtype hdr)))
      (is (= false (:fortran? hdr)))
      (is (= [10] (:shape hdr)))
      (is (pos-int? (:data-start hdr))))))

(deftest parse-npy-header-basic-2d
  (testing "parse-npy-header parses a 2D f4 fixture"
    (let [bs  (sfs/slurp-bytes (resource-path "shape_2x3__dtype_f4.npy"))
          hdr (parse-npy-header bs)]
      (is (= "<f4" (:descr hdr)))
      (is (= :little (:byte-order hdr)))
      (is (= "f" (:kind hdr)))
      (is (= 4 (:size hdr)))
      (is (= :f4 (:dtype hdr)))
      (is (= false (:fortran? hdr)))
      (is (= [2 3] (:shape hdr)))
      (is (pos-int? (:data-start hdr))))))

(deftest parse-npy-header-int64
  (testing "parse-npy-header parses i8 correctly"
    (let [bs  (sfs/slurp-bytes (resource-path "shape_2x3__dtype_i8.npy"))
          hdr (parse-npy-header bs)]
      (is (= "<i8" (:descr hdr)))
      (is (= :little (:byte-order hdr)))
      (is (= "i" (:kind hdr)))
      (is (= 8 (:size hdr)))
      (is (= :i8 (:dtype hdr)))
      (is (= [2 3] (:shape hdr))))))

(deftest parse-npy-header-data-start-consistency
  (testing ":data-start lands at the end of the header"
    (let [bs (sfs/slurp-bytes (resource-path "shape_2x3__dtype_u2.npy"))
          ;; if you have read-header-string or read-header-len exposed internally,
          ;; skip this test; otherwise just do a basic sanity bound check:
          hdr (parse-npy-header bs)
          ds  (:data-start hdr)]
      ;; data-start should be within the file and leave at least 1 byte of payload
      (is (<= 0 ds (dec (alength bs))))
      ;; for our fixtures, the payload is non-empty
      (is (< ds (alength bs))))))

(deftest parse-npy-header-fortran-order-u4
  (testing "parse-npy-header parses fortran_order=True for u4 fixtures"
    (doseq [[shape fname] [[[2 3]   "shape_2x3__dtype_u4__order_F.npy"]
                           [[10 12] "shape_10x12__dtype_u4__order_F.npy"]]]
      (let [bs  (sfs/slurp-bytes (resource-path fname))
            hdr (parse-npy-header bs)]
        (is (= "<u4" (:descr hdr)) (str "descr mismatch for " fname))
        (is (= :little (:byte-order hdr)) (str "byte-order mismatch for " fname))
        (is (= "u" (:kind hdr)) (str "kind mismatch for " fname))
        (is (= 4 (:size hdr)) (str "size mismatch for " fname))
        (is (= :u4 (:dtype hdr)) (str "dtype mismatch for " fname))
        (is (= true (:fortran? hdr)) (str "fortran? mismatch for " fname))
        (is (= shape (:shape hdr)) (str "shape mismatch for " fname))
        (is (pos-int? (:data-start hdr)) (str "data-start invalid for " fname))))))

(deftest parse-npy-header-big-endian
  (testing "parse-npy-header parses big-endian dtypes"
    (doseq [[fname descr dtype kind size]
            [["shape_2x3__dtype_i2__endian_B.npy" ">i2" :i2 "i" 2]
             ["shape_2x3__dtype_u4__endian_B.npy" ">u4" :u4 "u" 4]
             ["shape_2x3__dtype_f8__endian_B.npy" ">f8" :f8 "f" 8]]]
      (let [bs  (sfs/slurp-bytes (resource-path fname))
            hdr (parse-npy-header bs)]
        (is (= descr (:descr hdr)) (str "descr mismatch for " fname))
        (is (= :big (:byte-order hdr)) (str "byte-order mismatch for " fname))
        (is (= kind (:kind hdr)) (str "kind mismatch for " fname))
        (is (= size (:size hdr)) (str "size mismatch for " fname))
        (is (= dtype (:dtype hdr)) (str "dtype mismatch for " fname))
        (is (= false (:fortran? hdr)) (str "fortran? mismatch for " fname))
        (is (= [2 3] (:shape hdr)) (str "shape mismatch for " fname))
        (is (pos-int? (:data-start hdr)) (str "data-start invalid for " fname))))))

(ns sturdy.numpy.magic-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [sturdy.fs :as sfs]
   [sturdy.numpy.test-utils :refer [resource-path]]
   [sturdy.numpy.magic :refer [read-header-string]]))

(deftest read-header-string-smoke
  (testing "read-header-string returns the expected header contents (up to whitespace)"
    (let [bs   (sfs/slurp-bytes (resource-path "shape_2x3__dtype_f4.npy"))
          {:keys [header-string]} (read-header-string bs)]
      ;; check the three essential fields are present
      (is (re-find #"'descr'\s*:\s*'<f4'" header-string))
      (is (re-find #"'fortran_order'\s*:\s*False" header-string))
      (is (re-find #"'shape'\s*:\s*\(\s*2\s*,\s*3\s*\)" header-string)))))

(deftest read-header-string-1d-shape
  (testing "read-header-string for a 1D u4 array"
    (let [bs  (sfs/slurp-bytes (resource-path "shape_10___dtype_u4.npy"))
          {:keys [header-string]} (read-header-string bs)]
      (is (re-find #"'descr'\s*:\s*'<u4'" header-string))
      (is (re-find #"'shape'\s*:\s*\(\s*10\s*,\s*\)" header-string)))))

(deftest read-header-string-int32
  (testing "read-header-string for i4"
    (let [bs  (sfs/slurp-bytes (resource-path "shape_2x3__dtype_i4.npy"))
          {:keys [header-string]} (read-header-string bs)]
      (is (re-find #"'descr'\s*:\s*'<i4'" header-string))
      (is (re-find #"'shape'\s*:\s*\(\s*2\s*,\s*3\s*\)" header-string)))))

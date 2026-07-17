(ns sturdy.numpy-test
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [tech.v3.dataset :as ds]
            [sturdy.fs :as sfs]
            [sturdy.numpy :as sn]
            [sturdy.numpy.test-utils :refer [resource-path]])
  (:import
   (java.nio.file Files)))

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

(deftest public-api-max-file-bytes-test
  (let [path      (resource-path "shape_2x3__dtype_i4.npy")
        file-size (Files/size (.toPath (io/file path)))
        readers   [["npy->vec" sn/npy->vec]
                   ["npy->dataset" sn/npy->dataset]
                   ["npy->primitive" sn/npy->primitive]
                   ["npy->dataset-rowlists" sn/npy->dataset-rowlists]
                   ["npy->dataset-unnested-nz" sn/npy->dataset-unnested-nz]]]
    (testing "the exact file-size boundary is accepted by every public reader"
      (doseq [[label reader] readers]
        (is (some? (reader path {:max-file-bytes file-size})) label)))

    (testing "a file above the limit is rejected before reading"
      (doseq [[label reader] readers]
        (let [error (try
                      (reader path {:max-file-bytes (dec file-size)})
                      nil
                      (catch clojure.lang.ExceptionInfo e
                        e))]
          (is (= "NumPy file exceeds configured size limit" (ex-message error)) label)
          (is (= {:actual file-size
                  :maximum (dec file-size)
                  :limit :max-file-bytes
                  :phase :before-read}
                 (select-keys (ex-data error)
                              [:actual :maximum :limit :phase]))
              label))))

    (testing "growth between the pre-read and post-read checks is rejected"
      (let [error (with-redefs [sfs/slurp-bytes
                                (fn [_] (byte-array (inc file-size)))]
                    (try
                      (sn/npy->primitive path {:max-file-bytes file-size})
                      nil
                      (catch clojure.lang.ExceptionInfo e
                        e)))]
        (is (= "NumPy file exceeds configured size limit" (ex-message error)))
        (is (= {:actual (inc file-size)
                :maximum file-size
                :limit :max-file-bytes
                :phase :after-read}
               (select-keys (ex-data error)
                            [:actual :maximum :limit :phase])))))

    (testing "unknown options are rejected"
      (let [error (try
                    (sn/npy->primitive path {:max-file-byte file-size})
                    nil
                    (catch clojure.lang.ExceptionInfo e
                      e))]
        (is (= "Unknown NumPy reader options" (ex-message error)))
        (is (= #{:max-file-byte} (:unknown-options (ex-data error))))))

    (testing "invalid limits are rejected"
      (doseq [value [-1 1.5 "100" (inc (bigint Long/MAX_VALUE))]]
        (let [error (try
                      (sn/npy->primitive path {:max-file-bytes value})
                      nil
                      (catch clojure.lang.ExceptionInfo e
                        e))]
          (is (= "Invalid :max-file-bytes option" (ex-message error))
              (str "value " (pr-str value)))
          (is (= value (:value (ex-data error)))))))))

(ns sturdy.numpy.dtype-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [sturdy.numpy.dtype :refer [dtype->bytes+reader]]))

(set! *warn-on-reflection* true)

(defn- sizing-error [size shape]
  (try
    (dtype->bytes+reader {:dtype :i4
                          :byte-order :little
                          :size size
                          :shape shape})
    nil
    (catch clojure.lang.ExceptionInfo e
      e)))

(deftest dtype-sizing-supports-empty-arrays
  (doseq [shape [[0] [0 3] [4 0] [0 0]]]
    (testing (str "shape " shape)
      (is (= 0 (:nbytes (dtype->bytes+reader {:dtype :i4
                                               :byte-order :little
                                               :size 4
                                               :shape shape})))))))

(deftest dtype-sizing-rejects-invalid-or-unrepresentable-shapes
  (testing "negative dimensions"
    (let [error (sizing-error 4 [2 -1])]
      (is (= "Invalid .npy shape dimension" (ex-message error)))
      (is (= {:shape [2 -1] :dimension -1 :reason :negative}
             (select-keys (ex-data error) [:shape :dimension :reason])))))

  (testing "a dimension beyond Java array indexing"
    (let [dimension (inc (long Integer/MAX_VALUE))
          error     (sizing-error 1 [dimension 0])]
      (is (= "Unsupported .npy shape dimension" (ex-message error)))
      (is (= {:shape [dimension 0]
              :dimension dimension
              :maximum Integer/MAX_VALUE
              :reason :array-index-limit}
             (select-keys (ex-data error)
                          [:shape :dimension :maximum :reason])))))

  (testing "element-count multiplication overflow"
    (let [error (sizing-error 1 [Long/MAX_VALUE 2])]
      (is (= "Invalid .npy element count" (ex-message error)))
      (is (= :arithmetic-overflow (:reason (ex-data error))))))

  (testing "element count beyond Java array indexing"
    (let [error (sizing-error 1 [Integer/MAX_VALUE 2])]
      (is (= "Unsupported .npy element count" (ex-message error)))
      (is (= {:element-count (* (long Integer/MAX_VALUE) 2)
              :maximum Integer/MAX_VALUE
              :reason :array-index-limit}
             (select-keys (ex-data error)
                          [:element-count :maximum :reason])))))

  (testing "payload-byte multiplication overflow"
    (let [error (sizing-error Long/MAX_VALUE [2])]
      (is (= "Invalid .npy payload size" (ex-message error)))
      (is (= :arithmetic-overflow (:reason (ex-data error))))))

  (testing "payload size beyond Java array indexing"
    (let [element-count (inc (quot Integer/MAX_VALUE 4))
          error         (sizing-error 4 [element-count])]
      (is (= "Unsupported .npy payload size" (ex-message error)))
      (is (= {:nbytes (* 4 (long element-count))
              :maximum Integer/MAX_VALUE
              :reason :array-index-limit}
             (select-keys (ex-data error) [:nbytes :maximum :reason]))))))

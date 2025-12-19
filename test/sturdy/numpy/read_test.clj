(ns sturdy.numpy.read-test
  (:require
   [clojure.string :as string]
   [clojure.test :refer [deftest is testing]]
   [sturdy.numpy.test-utils :refer [resource-path]]
   [sturdy.numpy.read :refer [read-npy]]))

(defn- get-shape [vec-of-vecs]
  (let [rows (count vec-of-vecs)
        el   (first vec-of-vecs)
        cols (when (vector? el) (count el))]
    (if cols [rows cols] [rows])))

;; --- Expected value generators (must match make_npy_fixtures.py) ---

(defn expected-vals-1d
  "Generate the flat 1D expected sequence (length n) for a dtype keyword."
  [dtype n]
  (case dtype
    ;; floats: base = arange(n)*1.25 - 3.0, then cast
    :f4 (mapv #(float  (- (* 1.25 %) 3.0)) (range n))
    :f8 (mapv #(double (- (* 1.25 %) 3.0)) (range n))

    ;; ints/uints: base = arange(n)*17 - 5, then cast/wrap
    :i1 (mapv #(int (byte  (- (* 17 %) 5))) (range n))
    :i2 (mapv #(int (short (- (* 17 %) 5))) (range n))
    :i4 (mapv #(int  (- (* 17 %) 5)) (range n))
    :i8 (mapv #(long (- (* 17 %) 5)) (range n))

    ;; unsigned in our reader returns int-array/long-array then vec,
    ;; so expect non-negative numbers with wrapping.
    :u1 (mapv #(int (bit-and (int (- (* 17 %) 5)) 0xFF))   (range n))
    :u2 (mapv #(int (bit-and (int (- (* 17 %) 5)) 0xFFFF)) (range n))
    :u4 (mapv #(long (Integer/toUnsignedLong (int (- (* 17 %) 5)))) (range n))

    (throw (ex-info "Unsupported dtype in expected generator" {:dtype dtype}))))

(defn expected-data
  "Expected :data for given shape + dtype, row-major, matching Python generator."
  [shape dtype]
  (let [n (reduce * 1 shape)
        flat (expected-vals-1d dtype n)]
    (if (= 1 (count shape))
      flat
      (let [[rows cols] shape]
        (vec
         (for [r (range rows)]
           (subvec flat (* r cols) (* (inc r) cols))))))))

;; --- Tests ---

(deftest read-npy-dtypes-on-2x3
  (testing "Reads each dtype for shape (2,3)"
    (doseq [dtype [:u1 :u2 :u4 :i1 :i2 :i4 :i8 :f4 :f8]]
      (let [fname (format "shape_2x3__dtype_%s.npy" (name dtype))
            res   (read-npy (resource-path fname))]
        (is (= [2 3] (get-shape res)) (str "shape mismatch for " fname))
        (is (= (expected-data [2 3] dtype) res)
            (str "data mismatch for " fname))))))

(deftest read-npy-shapes-on-u4
  (testing "Reads each shape for dtype u4"
    (doseq [shape [[1] [10] [1 10] [2 3] [10 1] [10 12]]]
      (let [shape-tag (if (= 1 (count shape))
                        (format "%d_" (first shape))
                        (string/join "x" shape))
            fname (format "shape_%s__dtype_u4.npy" shape-tag)
            res   (read-npy (resource-path fname))]
        (is (= shape (get-shape res)) (str "shape mismatch for " fname))
        (is (= (expected-data shape :u4) res)
            (str "data mismatch for " fname))))))

(deftest read-npy-manual-small
  (testing "Manual fixture for eyeballing"
    (let [res (read-npy (resource-path "manual_2x3__dtype_i4.npy"))]
      (is (= [2 3] (get-shape res)))
      (is (= [[1 2 3] [4 5 6]] res)))))

(deftest read-npy-fortran-order-normalizes-to-row-major
  (testing "Fortran-order .npy is returned as row-major (C-order) data"
    (doseq [[shape fname] [[[2 3]   "shape_2x3__dtype_u4__order_F.npy"]
                           [[10 12] "shape_10x12__dtype_u4__order_F.npy"]]]
      (let [res (read-npy (resource-path fname))]
        (is (= shape (get-shape res)) (str "shape mismatch for " fname))
        ;; Contract: always materialize :data in row-major layout
        (is (= (expected-data shape :u4) res)
            (str "data mismatch for " fname))))))

(deftest read-npy-big-endian-normalizes-correctly
  (testing "Big-endian .npy is read correctly and returned as normal Clojure data"
    (doseq [[dtype fname]
            [[:i2 "shape_2x3__dtype_i2__endian_B.npy"]
             [:u4 "shape_2x3__dtype_u4__endian_B.npy"]
             [:f8 "shape_2x3__dtype_f8__endian_B.npy"]]]
      (let [res (read-npy (resource-path fname))]
        (is (= [2 3] (get-shape res)) (str "shape mismatch for " fname))
        (is (= (expected-data [2 3] dtype) res)
            (str "data mismatch for " fname))))))

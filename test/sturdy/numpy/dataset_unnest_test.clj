(ns sturdy.numpy.dataset-unnest-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [clojure.string :as str]
   [tech.v3.dataset :as ds]
   [tech.v3.datatype :as dtype]
   [sturdy.numpy.dataset-unnest :refer [npy->dataset-unnested-nz]]
   [sturdy.numpy.test-utils :refer [resource-path]]
   [sturdy.numpy.read-test :refer [expected-vals-1d]]))

;; ---- Shared expected helpers (reuse your existing logic) ----

(defn expected-cols
  "Return a vector of expected column vectors for given shape+dtype,
   assuming the file data is row-major (C-order) and we build a columnar dataset."
  [shape dtype]
  (let [n    (reduce * 1 shape)
        flat (expected-vals-1d dtype n)]
    (if (= 1 (count shape))
      [(vec flat)]
      (let [[rows cols] shape]
        (mapv (fn [j]
                (mapv (fn [r] (nth flat (+ (* r cols) j)))
                      (range rows)))
              (range cols))))))

(defn- shape->rows-cols [shape]
  (case (count shape)
    1 [(long (first shape)) 1]
    2 [(long (first shape)) (long (second shape))]
    (throw (ex-info "invalid shape (must be 1D or 2D)" {:shape shape}))))

;; ---- New helpers for the long / nz dataset ----

(defn- ds-col->vec [ds cname]
  (vec (get ds cname)))

(defn- long-ds->dense-cols
  "Given a long-form dataset with columns :row_no, :col_no, :val,
   reconstruct a dense column-major representation as a vector-of-columns,
   with missing entries treated as 0.

   Returns same shape as expected-cols: [col0 col1 ...], each col is a vector length rows."
  [shape dtype long-ds]
  (let [[rows cols] (shape->rows-cols shape)
        ;; Build mutable dense backing. Use object arrays for simplicity;
        ;; then cast to vectors at end. This is tests, not hot path.
        dense (vec (repeatedly cols #(object-array rows)))

        rowv (ds-col->vec long-ds :row_no)
        colv (ds-col->vec long-ds :col_no)
        valv (ds-col->vec long-ds :val)

        ;; Choose a 0 value of the right "Clojure level" numeric type.
        ;; expected-vals-1d will produce Longs for integer-ish, Doubles/Floats for float-ish.
        ;; For unsigned, we still compare via =, so 0 as long is fine.
        z (case dtype
            (:f4 :f8) 0.0
            0)]
    ;; initialize all to zero
    (doseq [j (range cols)]
      (let [^objects colarr (dense j)]
        (dotimes [r rows]
          (aset colarr r z))))
    ;; apply nonzeros
    (doseq [i (range (count valv))]
      (let [r (long (nth rowv i))
            c (int  (short (nth colv i)))  ;; stored as short; normalize
            v (nth valv i)]
        (aset ^objects (dense c) (int r) v)))
    ;; return as [col0 col1 ...] each a vector
    (mapv (fn [j] (vec (seq (dense j))))
          (range cols))))

(defn- expected-nnz
  "Count nonzeros in the expected dense data for shape+dtype."
  [shape dtype]
  (let [[_rows cols] (shape->rows-cols shape)
        exp (expected-cols shape dtype)]
    (reduce
     (fn [acc j]
       (let [col (exp j)]
         (+ acc (count (remove #(= % 0) col)))))
     0
     (range cols))))

;; ---- Tests ----

(deftest npy->dataset-unnested-nz-dtypes-on-2x3
  (testing "Builds a long/nz dataset with correct (row_no,col_no,val) for each dtype (shape 2x3)"
    (doseq [dtype [:u1 :u2 :u4 :i1 :i2 :i4 :i8 :f4 :f8]]
      (let [fname (format "shape_2x3__dtype_%s.npy" (name dtype))
            ds    (npy->dataset-unnested-nz (resource-path fname))
            exp   (expected-cols [2 3] dtype)
            got   (long-ds->dense-cols [2 3] dtype ds)]
        (is (= 3 (count (ds/column-names ds))) (str "column count mismatch for " fname))
        (is (= #{:row_no :col_no :val} (set (ds/column-names ds)))
            (str "missing columns for " fname))
        (is (= (expected-nnz [2 3] dtype) (ds/row-count ds))
            (str "nnz row count mismatch for " fname))
        (is (= exp got) (str "reconstructed dense mismatch for " fname))))))

(deftest npy->dataset-unnested-nz-shapes-on-u4
  (testing "Builds a long/nz dataset that reconstructs correctly for dtype u4"
    (doseq [shape [[1] [10] [1 10] [2 3] [10 1] [10 12]]]
      (let [shape-tag (if (= 1 (count shape))
                        (format "%d_" (first shape))
                        (str/join "x" shape))
            fname (format "shape_%s__dtype_u4.npy" shape-tag)
            ds    (npy->dataset-unnested-nz (resource-path fname))
            exp   (expected-cols shape :u4)
            got   (long-ds->dense-cols shape :u4 ds)]
        (is (= #{:row_no :col_no :val} (set (ds/column-names ds)))
            (str "missing columns for " fname))
        (is (= (expected-nnz shape :u4) (ds/row-count ds))
            (str "nnz row count mismatch for " fname))
        (is (= exp got) (str "reconstructed dense mismatch for " fname))))))

(deftest npy->dataset-unnested-nz-manual-small
  (testing "Manual fixture for eyeballing (2x3 i4)"
    (let [ds  (npy->dataset-unnested-nz (resource-path "manual_2x3__dtype_i4.npy"))
          got (long-ds->dense-cols [2 3] :i4 ds)]
      (is (= [[1 4] [2 5] [3 6]] got)))))

(deftest npy->dataset-unnested-nz-fortran-shapes-on-u4
  (testing "Fortran-order fixtures reconstruct identically (ordering in long table may differ)"
    (doseq [[shape fname] [[[2 3]   "shape_2x3__dtype_u4__order_F.npy"]
                           [[10 12] "shape_10x12__dtype_u4__order_F.npy"]]]
      (let [ds  (npy->dataset-unnested-nz (resource-path fname))
            exp (expected-cols shape :u4)
            got (long-ds->dense-cols shape :u4 ds)]
        (is (= (expected-nnz shape :u4) (ds/row-count ds))
            (str "nnz row count mismatch for " fname))
        (is (= exp got) (str "reconstructed dense mismatch for " fname))))))

(deftest npy->dataset-unnested-nz-big-endian-on-2x3
  (testing "Big-endian fixtures reconstruct correctly (shape 2x3)"
    (doseq [[dtype fname]
            [[:i2 "shape_2x3__dtype_i2__endian_B.npy"]
             [:u4 "shape_2x3__dtype_u4__endian_B.npy"]
             [:f8 "shape_2x3__dtype_f8__endian_B.npy"]]]
      (let [ds  (npy->dataset-unnested-nz (resource-path fname))
            exp (expected-cols [2 3] dtype)
            got (long-ds->dense-cols [2 3] dtype ds)]
        (is (= (expected-nnz [2 3] dtype) (ds/row-count ds))
            (str "nnz row count mismatch for " fname))
        (is (= exp got) (str "reconstructed dense mismatch for " fname))))))

(deftest npy->dataset-unnested-nz-preserves-column-dtypes
  (testing "Long/nz dataset has expected tech.datatype dtypes for row_no/col_no/val"
    (doseq [[tp expected-val]
            [[:u1 :uint8]
             [:u2 :uint16]
             [:u4 :uint32]
             [:i1 :int8]
             [:i2 :int16]
             [:i4 :int32]
             [:i8 :int64]
             [:f4 :float32]
             [:f8 :float64]]]
      (let [fname   (format "shape_2x3__dtype_%s.npy" (name tp))
            dataset (npy->dataset-unnested-nz (resource-path fname))
            row-col (get dataset :row_no)
            col-col (get dataset :col_no)
            val-col (get dataset :val)]
        (is (= :int64 (dtype/elemwise-datatype row-col))
            (str "row_no dtype mismatch for " fname))
        (is (= :int16 (dtype/elemwise-datatype col-col))
            (str "col_no dtype mismatch for " fname))
        (is (= expected-val (dtype/elemwise-datatype val-col))
            (str "val dtype mismatch for " fname " expected=" expected-val))))))

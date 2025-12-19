(ns sturdy.numpy.dataset-unnest
  (:require
   [sturdy.numpy.read :refer [read-npy-primitive]]
   [tech.v3.dataset :as ds]
   [tech.v3.datatype :as dtype]))

(set! *warn-on-reflection* true)

(defn- unsigned-val-dtype [dtype]
  (case dtype
    :u1 :uint8
    :u2 :uint16
    :u4 :uint32
    nil))

(defn- shape->rows-cols
  [shape]
  (case (count shape)
    1 [(long (first shape)) 1]
    2 [(long (first shape)) (long (second shape))]
    (throw (ex-info "invalid shape (must be 1D or 2D)" {:shape shape}))))

(defmacro def-unnester-skip-zeros
  "Define an unnest fn that skips val==0.
   Returns {:row_no long[] :col_no short[] :val <primitive[]>} sized to nnz.

   src-tag:  bytes, shorts, ints, longs, floats, doubles
   val-array-ctor: e.g. float-array
   aset-val: e.g. aset-float
   zero-lit: literal 0 of the right primitive type (0, 0.0, (byte 0), etc)."
  [fname src-tag val-array-ctor aset-val zero-lit]
  (let [src      (with-meta (gensym "src") {:tag src-tag})
        rows     (gensym "rows")
        cols     (gensym "cols")
        fortran? (gensym "fortran?")
        n        (gensym "n")
        nnz      (gensym "nnz")
        rowa     (gensym "rowa")
        cola     (gensym "cola")
        vala     (with-meta (gensym "vala") {:tag src-tag})]
    `(defn ~fname
       [~src ~rows ~cols ~fortran?]
       (let [~n (long (* (long ~rows) (long ~cols)))
             ;; pass 1: count nnz
             ~nnz (loop [k# 0
                         c# 0]
                    (if (< k# ~n)
                      (let [v# (aget ~src k#)]
                        (recur (long (inc k#))
                               (long (if (== v# ~zero-lit)
                                       c#
                                       (inc c#)))))
                      (long c#)))
             ;; allocate only nnz
             ~rowa (long-array ~nnz)
             ~cola (short-array ~nnz)
             ~vala (~val-array-ctor ~nnz)]
         ;; pass 2: fill
         (loop [k# 0
                out# 0]
           (when (< k# ~n)
             (let [v# (aget ~src k#)]
               (if (== v# ~zero-lit)
                 (recur (long (inc k#)) out#)
                 (let [kL# (long k#)
                       r#  (if ~fortran?
                             (long (rem kL# (long ~rows)))
                             (long (quot kL# (long ~cols))))
                       c#  (if ~fortran?
                             (long (quot kL# (long ~rows)))
                             (long (rem kL# (long ~cols))))]
                   (aset-long  ~rowa out# r#)
                   (aset-short ~cola out# (short c#))
                   (~aset-val  ~vala out# v#)
                   (recur (long (inc k#)) (long (inc out#))))))))
         {:row_no ~rowa
          :col_no ~cola
          :val    ~vala}))))

;; Define for each primitive type.
(def-unnester-skip-zeros unnest-float-nz  floats  float-array  aset-float  0.0)
(def-unnester-skip-zeros unnest-double-nz doubles double-array aset-double 0.0)

;; For integer types, (== v# 0) is fine. Use explicit casts for byte/short literals
;; just to keep everything primitive and avoid any surprises.
(def-unnester-skip-zeros unnest-byte-nz   bytes   byte-array   aset-byte   (byte 0))
(def-unnester-skip-zeros unnest-short-nz  shorts  short-array  aset-short  (short 0))
(def-unnester-skip-zeros unnest-int-nz    ints    int-array    aset-int    0)
(def-unnester-skip-zeros unnest-long-nz   longs   long-array   aset-long   0)

(defn dtype->unnester-nz [dtype]
  (cond
    (#{:f8} dtype)         unnest-double-nz
    (#{:f4} dtype)         unnest-float-nz

    (#{:i8 :u4} dtype)     unnest-long-nz
    (#{:i4 :u2} dtype)     unnest-int-nz
    (#{:i2 :u1} dtype)     unnest-short-nz
    (#{:i1} dtype)         unnest-byte-nz

    :else
    (throw (ex-info "unknown dtype" {:dtype dtype}))))

(defn npy->dataset-unnested-nz
  "Same as npy->dataset-unnested, but skips val==0 (2-pass)."
  [path]
  (let [{:keys [shape dtype fortran? data]} (read-npy-primitive path)
        [rows cols] (shape->rows-cols shape)
        _ (when (>= cols 32768)
            (throw (ex-info "cols too large for short col_no" {:cols cols})))

        ;; If 1D, treat as rows x 1 regardless of fortran?
        fortran? (boolean (and (= 2 (count shape)) fortran?))

        f (dtype->unnester-nz dtype)
        {:keys [row_no col_no val]} (f data rows cols fortran?)

        ;; Preserve unsigned representation for DuckDB ingest (like before).
        target (unsigned-val-dtype dtype)
        val'   (if target (dtype/->array-buffer target val) val)]
    (ds/->dataset {:row_no row_no
                   :col_no col_no
                   :val    val'})))

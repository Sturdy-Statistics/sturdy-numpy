(ns sturdy.numpy.dataset
  (:require
   [sturdy.numpy.read :refer [read-npy-primitive]]
   [tech.v3.dataset :as ds]
   [tech.v3.datatype :as dtype])
  (:import
   (java.util Arrays)))

(set! *warn-on-reflection* true)

;; For C-order payloads, transpose into column vectors.
;; For Fortran-order payloads, split directly into column vectors.

(defmacro def-transposer
  "Define a transpose fn for a primitive array type.
   src-tag is one of: bytes, shorts, ints, longs, floats, doubles
   array-ctor is e.g. float-array
   aset-fn is e.g. aset-float"
  [fname src-tag array-ctor aset-fn]
  (let [src  (with-meta (gensym "src")  {:tag src-tag})
        rows (gensym "rows")
        cols (gensym "cols")
        dsts (gensym "dsts")
        dstj (with-meta (gensym "dst") {:tag src-tag})]
    `(defn ~fname
       [~src ~rows ~cols]
       (let [~dsts (vec (repeatedly ~cols #(~array-ctor ~rows)))]
         (loop [i# 0
                base# 0]
           (when (< i# ~rows)
             (loop [j# 0]
               (when (< j# ~cols)
                 (let [~dstj (~dsts j#)]
                   (~aset-fn ~dstj i# (aget ~src (+ base# j#))))
                 (recur (long (inc j#)))))
             (recur (long (inc i#)) (long (+ base# ~cols)))))
         ~dsts))))

(def-transposer transpose-float  floats  float-array  aset-float)
(def-transposer transpose-double doubles double-array aset-double)
(def-transposer transpose-byte   bytes   byte-array   aset-byte)
(def-transposer transpose-short  shorts  short-array  aset-short)
(def-transposer transpose-int    ints    int-array    aset-int)
(def-transposer transpose-long   longs   long-array   aset-long)

(defn dtype->transposer [dtype]
  (cond
    (#{:f8} dtype)         transpose-double
    (#{:f4} dtype)         transpose-float

    (#{:i8 :u4} dtype)     transpose-long
    (#{:i4 :u2} dtype)     transpose-int
    (#{:i2 :u1} dtype)     transpose-short
    (#{:i1} dtype)         transpose-byte

    :else
    (throw (ex-info "unknown dtype" {:dtype dtype}))))

(defmacro def-splitter
  "Generate a function that splits a column-major flat primitive array
   into a vector of per-column primitive arrays (each length = rows)."
  [fname src-tag]
  (let [src  (with-meta (gensym "src")  {:tag src-tag})
        rows (gensym "rows")
        cols (gensym "cols")]
    `(defn ~fname
       [~src ~rows ~cols]
       (loop [j# 0
              off# 0
              acc# (transient [])]
         (if (< j# ~cols)
           (recur (long (inc j#))
                  (long (+ off# ~rows))
                  (conj! acc# (Arrays/copyOfRange ~src off# (long (+ off# ~rows)))))
           (persistent! acc#))))))

(def-splitter split-float  floats)
(def-splitter split-double doubles)
(def-splitter split-byte   bytes)
(def-splitter split-short  shorts)
(def-splitter split-int    ints)
(def-splitter split-long   longs)

(defn dtype->splitter [dtype]
  (cond
    (#{:f8} dtype)         split-double
    (#{:f4} dtype)         split-float

    (#{:i8 :u4} dtype)     split-long
    (#{:i4 :u2} dtype)     split-int
    (#{:i2 :u1} dtype)     split-short
    (#{:i1} dtype)         split-byte

    :else
    (throw (ex-info "unknown dtype" {:dtype dtype}))))

(defn- unsigned-col-dtype [dtype]
  (case dtype
    :u1 :uint8
    :u2 :uint16
    :u4 :uint32
    nil))

(defn- data->cols
  [{:keys [shape dtype fortran? data]}]
  (case (count shape)
    1 [data]
    2 (let [[rows cols] shape
            f (if fortran?
                (dtype->splitter dtype)
                (dtype->transposer dtype))]
        (f data rows cols))
    (throw (ex-info "invalid shape (must be 1D or 2D)" {:shape shape}))))

(defn npy->dataset
  [path]
  (let [{:keys [shape dtype _fortran? _data] :as spec} (read-npy-primitive path)

        _rows      (first shape)
        cols       (if (= 1 (count shape)) 1 (second shape))

        col-data0  (data->cols spec)
        target     (unsigned-col-dtype dtype)
        col-data   (if target
                     (mapv #(dtype/->array-buffer target %) col-data0)
                     col-data0)

        col-names  (mapv #(keyword (str "c" (inc %)))
                         (range cols))

        result     (ds/->dataset (zipmap col-names col-data))]

    (ds/select-columns result col-names)))

(ns sturdy.numpy.read
  (:require
   [sturdy.fs :as sfs]
   [sturdy.numpy.util :refer [slice]]
   [sturdy.numpy.header :refer [parse-npy-header]]
   [sturdy.numpy.dtype :refer [dtype->bytes+reader]]))

(defn- read-values
  "Decode payload bytes into a flat Java primitive array."
  [^bytes bs data-start {:keys [nbytes reader]}]
  (let [payload (slice bs data-start (+ data-start nbytes))]
    (reader payload)))

(defn- array->vec1d
  "Convert a flat primitive array to a Clojure vector."
  [arr]
  (vec arr))

(defn- array->vec2d
  "Convert a flat primitive array to a vector-of-vectors.

   `order` is `:c` (row-major) or `:f` (column-major).
   The returned nested vectors are always in row-major order."
  [arr rows cols order]
  (let [pos (case order
              :c (fn [^long r ^long c] (+ (* r cols) c))
              :f (fn [^long r ^long c] (+ r (* rows c)))
              (throw (ex-info "Unsupported array order" {:order order})))]
    (vec
     (for [r (range rows)]
       (vec
        (for [c (range cols)]
          ;; Reflection here is expected: arr may be any primitive array type.
          (aget arr (pos r c))))))))

(defn- array->vec
  [arr shape fortran?]
  (if (= 1 (count shape))
    (array->vec1d arr)
    (let [[rows cols] shape
          order (if fortran? :f :c)]
      (array->vec2d arr rows cols order))))

(defn read-npy-primitive
  "Read a NumPy `.npy` file and return decoded values in a flat primitive array.

   Returns {:shape :dtype :fortran? :data} where `:data` is laid out according
   to `:fortran?` and no transposition or reshaping is performed."
  [path]
  (let [bs  (sfs/slurp-bytes path)
        hdr (parse-npy-header bs)
        {:keys [shape fortran? data-start] :as _hdr} hdr
        spec (dtype->bytes+reader hdr)
        arr  (read-values bs data-start spec)]
    {:shape    shape
     :dtype    (:dtype hdr)
     :fortran? fortran?
     :data     arr}))

(defn read-npy
  "Read a NumPy `.npy` file and return its contents as Clojure data.

   Returns a vector (1D) or vector of vectors (2D), always in row-major order."
  [path]
  (let [{:keys [shape fortran? data]} (read-npy-primitive path)]
    (array->vec data shape fortran?)))

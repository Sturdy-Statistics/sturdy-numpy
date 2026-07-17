(ns sturdy.numpy.read
  (:require
   [babashka.fs :as fs]
   [sturdy.fs :as sfs]
   [sturdy.numpy.util :refer [slice]]
   [sturdy.numpy.header :refer [parse-npy-header]]
   [sturdy.numpy.dtype :refer [dtype->bytes+reader]]))

(set! *warn-on-reflection* true)

(def ^:private allowed-options #{:max-file-bytes})

(defn- validate-options [options]
  (let [options (or options {})]
    (when-not (map? options)
      (throw (ex-info "NumPy reader options must be a map"
                      {:options options})))
    (let [unknown-options (set (remove allowed-options (keys options)))]
      (when (seq unknown-options)
        (throw (ex-info "Unknown NumPy reader options"
                        {:unknown-options unknown-options}))))
    (let [max-file-bytes (:max-file-bytes options)]
      (when (and (some? max-file-bytes)
                 (not (and (integer? max-file-bytes)
                           (<= 0 max-file-bytes Long/MAX_VALUE))))
        (throw (ex-info "Invalid :max-file-bytes option"
                        {:value max-file-bytes}))))
    options))

(defn- enforce-file-size-limit [actual maximum phase]
  (when (and (some? maximum) (> actual maximum))
    (throw (ex-info "NumPy file exceeds configured size limit"
                    {:actual actual
                     :maximum maximum
                     :limit :max-file-bytes
                     :phase phase}))))

(defn- read-values
  "Decode payload bytes into a flat Java primitive array."
  [^bytes bs data-start {:keys [nbytes reader]}]
  (let [data-end (try
                   (Math/addExact (long data-start) (long nbytes))
                   (catch ArithmeticException cause
                     (throw (ex-info "Invalid .npy data offset"
                                     {:data-start data-start
                                      :nbytes nbytes
                                      :reason :arithmetic-overflow}
                                     cause))))
        available (- (alength bs) data-start)]
    (when-not (= nbytes available)
      (throw (ex-info "Invalid .npy payload size"
                      {:expected nbytes
                       :available available})))
    (let [payload (slice bs data-start data-end)]
      (reader payload))))

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
          (java.lang.reflect.Array/get arr (pos r c))))))))

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
  ([path]
   (read-npy-primitive path nil))
  ([path options]
   (let [{:keys [max-file-bytes]} (validate-options options)
         _ (when (some? max-file-bytes)
             (enforce-file-size-limit (fs/size path)
                                      max-file-bytes
                                      :before-read))
         bs (sfs/slurp-bytes path)
         _ (enforce-file-size-limit (alength ^bytes bs)
                                    max-file-bytes
                                    :after-read)
         hdr (parse-npy-header bs)
         {:keys [shape fortran? data-start] :as _hdr} hdr
         spec (dtype->bytes+reader hdr)
         arr (read-values bs data-start spec)]
     {:shape    shape
      :dtype    (:dtype hdr)
      :fortran? fortran?
      :data     arr})))

(defn read-npy
  "Read a NumPy `.npy` file and return its contents as Clojure data.

   Returns a vector (1D) or vector of vectors (2D), always in row-major order."
  ([path]
   (read-npy path nil))
  ([path options]
   (let [{:keys [shape fortran? data]} (read-npy-primitive path options)]
     (array->vec data shape fortran?))))

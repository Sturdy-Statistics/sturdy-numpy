(ns sturdy.numpy.dtype
  (:import
   (java.nio ByteBuffer ByteOrder)))

(set! *warn-on-reflection* true)

(defn- bo->nio ^ByteOrder [bo]
  (case bo
    :little ByteOrder/LITTLE_ENDIAN
    :big    ByteOrder/BIG_ENDIAN
    ;; '|' means "not applicable" (byte-sized types); endianness irrelevant
    :na     ByteOrder/LITTLE_ENDIAN))

(defn- decode-into
  "Decode byte[] into a primitive array based on `:dtype` keyword.

   Unsigned dtypes are widened to the next signed primitive type:
   `:u1` -> short[], `:u2` -> int[], `:u4` -> long[]."
  [^bytes bs dtype byte-order]
  (let [bb (doto (ByteBuffer/wrap bs) (.order (bo->nio byte-order)))]
    (case dtype
      ;; floats
      :f8 (let [n (quot (alength bs) 8)
                arr (double-array n)]
            (dotimes [i n] (aset-double arr i (.getDouble bb)))
            arr)

      :f4 (let [n (quot (alength bs) 4)
                arr (float-array n)]
            (dotimes [i n] (aset-float arr i (.getFloat bb)))
            arr)

      ;; integers
      :i8 (let [n (quot (alength bs) 8)
                arr (long-array n)]
            (dotimes [i n] (aset-long arr i (.getLong bb)))
            arr)

      :i4 (let [n (quot (alength bs) 4)
                arr (int-array n)]
            (dotimes [i n] (aset-int arr i (.getInt bb)))
            arr)

      :i2 (let [n (quot (alength bs) 2)
                arr (short-array n)]
            (dotimes [i n] (aset-short arr i (.getShort bb)))
            arr)

      :i1 (byte-array bs)

      ;; uints. NB: clojure has no primitive unsigned arrays;
      ;; widen size by one step to recover the full range
      :u4 (let [n (quot (alength bs) 4)
                arr (long-array n)]
            (dotimes [i n] (aset-long arr i (Integer/toUnsignedLong (.getInt bb))))
            arr)

      :u2 (let [n (quot (alength bs) 2)
                arr (int-array n)]
            (dotimes [i n] (aset-int arr i (bit-and (int (.getShort bb)) 0xffff)))
            arr)

      :u1 (let [n (alength bs)
                arr (short-array n)]
            (dotimes [i n] (aset-short arr i (short (bit-and (int (.get bb)) 0xff))))
            arr)

      (throw (ex-info "Unsupported reader" {:dtype dtype})))))

(defn dtype->bytes+reader
  "Given parsed header fields like:
   {:descr \"<f4\" :byte-order :little :kind \"f\" :size 4}
   return {:dtype <keyword>
           :size <int>
           :shape <tuple>
           :nbytes <int>
           :reader (fn [bytes] ...)}.

   Reader returns a Java primitive array

   Supports: f4,f8, i1,i2,i4,i8, u1,u2,u4."
  [{:keys [dtype byte-order size shape]}]
  {:dtype   dtype
   :size    size
   :shape   shape
   :nbytes  (* size (reduce * 1 shape))
   :reader  (fn [^bytes payload]
              (decode-into payload dtype byte-order))})

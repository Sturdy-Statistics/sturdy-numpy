(ns sturdy.numpy.endian
  (:import
   (java.nio ByteBuffer ByteOrder)))

(set! *warn-on-reflection* true)

(defn uint16-le->long
  "Decode 2 LE bytes to a Java long in [0, 2^16-1]."
  ^long [^bytes ba]
  (when (not= 2 (alength ba))
    (throw (ex-info "uint16-le->long requires exactly 2 bytes" {})))
  (let [bb (doto (ByteBuffer/wrap ba) (.order ByteOrder/LITTLE_ENDIAN))
        s  (.getShort bb)]
    (bit-and (long s) 0xffff)))

(defn uint32-le->long
  "Decode 4 LE bytes to a Java long in [0, 2^32-1]."
  ^long [^bytes ba]
  (when (not= 4 (alength ba))
    (throw (ex-info "uint32-le->long requires exactly 4 bytes" {})))
  (let [bb (doto (ByteBuffer/wrap ba) (.order ByteOrder/LITTLE_ENDIAN))]
    (Integer/toUnsignedLong (.getInt bb))))

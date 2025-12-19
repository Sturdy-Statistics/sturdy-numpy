(ns sturdy.numpy.util
  (:import
   (java.util Arrays)))

(set! *warn-on-reflection* true)

(defn u8
  "Interpret a signed Java byte as an unsigned value in [0,255]."
  [b]
  (bit-and (int b) 0xff))

(defn slice ^bytes [^bytes bs ^long start ^long end]
  (Arrays/copyOfRange ^bytes bs ^long start ^long end))

(defn byte->hex-string [b]
  (format "%02x" (u8 b)))

(defn bytes->hex-string [^bytes bs]
  (apply str (map byte->hex-string bs)))

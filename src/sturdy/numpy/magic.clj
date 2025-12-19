(ns sturdy.numpy.magic
  (:require
   [clojure.string :as string]
   [sturdy.numpy.endian :as e]
   [sturdy.numpy.util :as u :refer [u8 slice]])
  (:import
   (java.util Arrays)))

(set! *warn-on-reflection* true)

(defn- check-magic
  [^bytes npy-byte-data]
  (when (< (alength npy-byte-data) 10)
    (throw (ex-info "Truncated .npy file" {:nbytes (alength npy-byte-data)})))

  (let [magic (slice npy-byte-data 0 6)
        major (aget npy-byte-data 6)
        minor (aget npy-byte-data 7)

        first-byte      (u8 (aget ^bytes magic 0))
        expected-bytes  (.getBytes "NUMPY" "US-ASCII")
        found-bytes     (slice magic 1 6)]

    (when-not (= (u8 0x93) first-byte)
      (throw (ex-info "invalid magic byte"
                      {:expected "0x93"
                       :found (format "0x%02x" first-byte)})))

    (when-not (Arrays/equals
               ^bytes expected-bytes
               ^bytes found-bytes)
      (throw (ex-info "invalid magic string"
                      {:expected (u/bytes->hex-string expected-bytes)
                       :found (u/bytes->hex-string found-bytes)})))

    {:major (u8 major)
     :minor (u8 minor)}))

(defn- read-header-len
  [^bytes npy-byte-data {:keys [major]}]
  (let [ofst    8
        size    (case (long major)
                  1 2
                  2 4
                  3 4
                (throw (ex-info "Unsupported .npy major version" {:major major})))
        data    (slice npy-byte-data ofst (+ ofst size))
        hdrlen  (case (long major)
                  1 (e/uint16-le->long data)
                  2 (e/uint32-le->long data)
                  3 (e/uint32-le->long data))]
    {:header-length hdrlen
     :header-start (+ ofst size)}))

(defn read-header-string
  [^bytes npy-byte-data]
  (let [ver    (check-magic npy-byte-data)

        {:keys [header-start header-length]}
        (read-header-len npy-byte-data ver)

        data-start (+ header-start header-length)

        hdr-bytes
        (slice npy-byte-data
               header-start
               data-start)

        hdr-string
        (String. ^bytes hdr-bytes "US-ASCII")]
    {:data-start data-start
     :header-string (string/trimr hdr-string)}))

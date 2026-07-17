(ns sturdy.numpy.magic
  (:require
   [clojure.string :as string]
   [sturdy.numpy.endian :as e]
   [sturdy.numpy.util :as u :refer [u8 slice]])
  (:import
   (java.nio.charset Charset StandardCharsets)
   (java.util Arrays)))

(set! *warn-on-reflection* true)

(defn- check-magic
  [^bytes npy-byte-data]
  (when (< (alength npy-byte-data) 8)
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

(defn- version-spec [{:keys [major]}]
  ;; Minor versions deliberately do not affect layout. No nonzero minor has
  ;; been specified, and accepting one is forward-compatible within a major.
  (case (long major)
    1 {:length-size 2 :charset StandardCharsets/ISO_8859_1}
    2 {:length-size 4 :charset StandardCharsets/ISO_8859_1}
    3 {:length-size 4 :charset StandardCharsets/UTF_8}
    (throw (ex-info "Unsupported .npy major version" {:major major}))))

(defn- read-header-len
  [^bytes npy-byte-data {:keys [major]} {:keys [length-size]}]
  (let [ofst      8
        available (- (alength npy-byte-data) ofst)]
    (when (< available length-size)
      (throw (ex-info "Truncated .npy header length"
                      {:expected length-size
                       :available available})))
    (let [data   (slice npy-byte-data ofst (+ ofst length-size))
          hdrlen (case (long major)
                   1 (e/uint16-le->long data)
                   2 (e/uint32-le->long data)
                   3 (e/uint32-le->long data))]
      {:header-length hdrlen
       :header-start (+ ofst length-size)})))

(defn read-header-string
  [^bytes npy-byte-data]
  (let [ver (check-magic npy-byte-data)
        {:keys [charset _length-size] :as spec} (version-spec ver)

        {:keys [header-start header-length]}
        (read-header-len npy-byte-data ver spec)

        data-start
        (try
          (Math/addExact (long header-start) (long header-length))
          (catch ArithmeticException cause
            (throw (ex-info "Invalid .npy header end"
                            {:header-start header-start
                             :header-length header-length
                             :reason :arithmetic-overflow}
                            cause))))

        available (- (alength npy-byte-data) header-start)

        _ (when (< available header-length)
            (throw (ex-info "Truncated .npy header"
                            {:expected header-length
                             :available available})))

        hdr-bytes
        (slice npy-byte-data
               header-start
               data-start)

        hdr-string
        (String. ^bytes hdr-bytes ^Charset charset)]
    {:data-start data-start
     :header-string (string/trimr hdr-string)}))

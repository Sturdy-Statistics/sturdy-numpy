(ns sturdy.numpy.header
  (:require
   [clojure.string :as string]
   [sturdy.numpy.magic :refer [read-header-string]]))

(set! *warn-on-reflection* true)

(def ^:private supported-dtypes
  #{:u1 :u2 :u4
    :i1 :i2 :i4 :i8
    :f4 :f8})

(defn- unsupported-descr [descr reason]
  (throw (ex-info "Unsupported .npy dtype descriptor"
                  {:descr descr :reason reason})))

(defn- parse-descr-value [^String descr-value]
  (let [[_ endian kind size-text]
        (or (re-matches #"^([<>|])([uif])([0-9]+)$" descr-value)
            (unsupported-descr descr-value :malformed))

        size
        (try
          (Long/parseLong size-text)
          (catch NumberFormatException _
            (unsupported-descr descr-value :malformed)))

        dtype (keyword (str kind size))]

    (when (and (= "|" endian) (not= 1 size))
      (unsupported-descr descr-value :invalid-byte-order))

    (let [byte-order
          (case endian
            "<" :little
            ">" :big
            "|" :na)]

      {:byte-order byte-order
       :kind       kind
       :size       size
       :dtype      dtype})))

(defn- parse-descr [^String hdr]
  ;; e.g. '<f4', '|u1', '>i8'
  (let [m (re-find #"[\"']descr[\"']\s*:\s*[\"']([^\"']*)[\"']" hdr)]
    (when-not m
      (throw (ex-info "Missing 'descr' in header" {:header hdr})))
    (let [descr-value (second m)

          {:keys [byte-order kind size dtype]}
          (parse-descr-value descr-value)]

      (when-not (supported-dtypes dtype)
        (unsupported-descr descr-value :unsupported-dtype))

      {:descr      descr-value
       :byte-order byte-order
       :kind       kind
       :size       size
       :dtype      dtype})))

(defn- parse-fortran-order [^String hdr]
  (let [m (re-find #"[\"']fortran_order[\"']\s*:\s*(True|False)" hdr)]
    (when-not m
      (throw (ex-info "Missing 'fortran_order' in header" {:header hdr})))
    (= "True" (second m))))

(defn- parse-shape [^String hdr]
  ;; Accepts (n,) or (r, c). Rejects anything else.
  (let [m (re-find #"[\"']shape[\"']\s*:\s*\(([^)]*)\)" hdr)]
    (when-not m
      (throw (ex-info "Missing 'shape' in header" {:header hdr})))
    (let [inside (-> (second m) string/trim)
          parts  (->> (string/split inside #",")
                      (map string/trim)
                      (remove empty?))
          dims   (mapv #(Long/parseLong %) parts)]
      (when-not (or (= 1 (count dims)) (= 2 (count dims)))
        (throw (ex-info "Only 1D/2D shapes supported" {:shape dims :header hdr})))
      (doseq [dimension dims]
        (when (neg? dimension)
          (throw (ex-info "Invalid .npy shape dimension"
                          {:shape dims
                           :dimension dimension
                           :reason :negative}))))
      dims)))

(defn parse-npy-header
  "Parse a NumPy .npy header string like:
   \"{'descr': '<f4', 'fortran_order': False, 'shape': (1,),}\"

   Returns:
   {:descr \"<f4\"
    :byte-order :little
    :kind \"f\"
    :size 4
    :dtype :f4
    :fortran? false
    :data-start int
    :shape [1]}"
  [^bytes npy-byte-data]
  (let [{:keys [header-string data-start]}
        (read-header-string npy-byte-data)

        hdr       (string/trim header-string)
        descr     (parse-descr hdr)
        fortran?  (parse-fortran-order hdr)
        shape     (parse-shape hdr)]
    (merge descr
           {:fortran? fortran?
            :shape shape
            :data-start data-start})))

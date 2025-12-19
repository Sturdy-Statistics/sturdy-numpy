(ns sturdy.numpy.header
  (:require
   [clojure.string :as string]
   [sturdy.numpy.magic :refer [read-header-string]]))

(set! *warn-on-reflection* true)

(defn- parse-descr [^String hdr]
  ;; e.g. '<f4', '|u1', '>i8'
  (let [m (re-find #"[\"']descr[\"']\s*:\s*[\"']([^\"']+)[\"']" hdr)]
    (when-not m
      (throw (ex-info "Missing 'descr' in header" {:header hdr})))
    (let [descr  (second m)
          endian (subs descr 0 1)
          kind   (subs descr 1 2)
          size   (Long/parseLong (subs descr 2))
          dtype  (keyword (str kind size))]
      (when-not (#{"u" "i" "f"} kind)
        (throw (ex-info "Unsupported dtype kind in descr" {:descr descr :kind kind})))
      {:descr descr
       :byte-order (case endian
                     "<" :little
                     ">" :big
                     "|" :na
                     (throw (ex-info "Unknown endianness in descr" {:descr descr})))
       :kind kind
       :size size
       :dtype dtype})))

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

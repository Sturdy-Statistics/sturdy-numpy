(ns sturdy.numpy.magic-test
  (:require
   [clojure.string :as string]
   [clojure.test :refer [deftest is testing]]
   [sturdy.fs :as sfs]
   [sturdy.numpy.test-utils :refer [resource-path]]
   [sturdy.numpy.magic :refer [read-header-string]])
  (:import
   (java.io ByteArrayOutputStream)
   (java.nio ByteBuffer ByteOrder)
   (java.nio.charset Charset StandardCharsets)
   (java.util Arrays)))

(set! *warn-on-reflection* true)

(defn- npy-bytes
  ([major minor header charset]
   (npy-bytes major minor header charset nil))
  ([major minor header ^Charset charset declared-length]
   (let [header-bytes (.getBytes ^String header charset)
         header-length (long (or declared-length (alength header-bytes)))
         length-size (if (= 1 major) 2 4)
         length-buffer (doto (ByteBuffer/allocate length-size)
                         (.order ByteOrder/LITTLE_ENDIAN))
         magic (.getBytes "\u0093NUMPY" StandardCharsets/ISO_8859_1)]
     (if (= 1 major)
       (.putShort length-buffer (short header-length))
       (.putInt length-buffer (int header-length)))
     (with-open [out (ByteArrayOutputStream.)]
       (.write out ^bytes magic 0 (alength ^bytes magic))
       (.write out (int major))
       (.write out (int minor))
       (let [length-bytes (.array length-buffer)]
         (.write out ^bytes length-bytes 0 (alength ^bytes length-bytes)))
       (.write out ^bytes header-bytes 0 (alength ^bytes header-bytes))
       (.toByteArray out)))))

(defn- header-error [^bytes bs]
  (try
    (read-header-string bs)
    nil
    (catch clojure.lang.ExceptionInfo e
      e)))

(deftest read-header-string-smoke
  (testing "read-header-string returns the expected header contents (up to whitespace)"
    (let [bs   (sfs/slurp-bytes (resource-path "shape_2x3__dtype_f4.npy"))
          {:keys [header-string]} (read-header-string bs)]
      ;; check the three essential fields are present
      (is (re-find #"'descr'\s*:\s*'<f4'" header-string))
      (is (re-find #"'fortran_order'\s*:\s*False" header-string))
      (is (re-find #"'shape'\s*:\s*\(\s*2\s*,\s*3\s*\)" header-string)))))

(deftest read-header-string-1d-shape
  (testing "read-header-string for a 1D u4 array"
    (let [bs  (sfs/slurp-bytes (resource-path "shape_10___dtype_u4.npy"))
          {:keys [header-string]} (read-header-string bs)]
      (is (re-find #"'descr'\s*:\s*'<u4'" header-string))
      (is (re-find #"'shape'\s*:\s*\(\s*10\s*,\s*\)" header-string)))))

(deftest read-header-string-int32
  (testing "read-header-string for i4"
    (let [bs  (sfs/slurp-bytes (resource-path "shape_2x3__dtype_i4.npy"))
          {:keys [header-string]} (read-header-string bs)]
      (is (re-find #"'descr'\s*:\s*'<i4'" header-string))
      (is (re-find #"'shape'\s*:\s*\(\s*2\s*,\s*3\s*\)" header-string)))))

(deftest read-header-string-uses-version-specific-charset
  (doseq [[major minor charset note]
          [[1 0 StandardCharsets/ISO_8859_1 "café"]
           [2 0 StandardCharsets/ISO_8859_1 "café"]
           ;; A nonzero minor is deliberately accepted; layout follows major 3.
           [3 47 StandardCharsets/UTF_8 "café π"]]]
    (testing (str "format version " major "." minor)
      (let [header (str "{'descr': '<i4', 'fortran_order': False, "
                        "'shape': (0,), 'note': '" note "'}\n")
            result (read-header-string (npy-bytes major minor header charset))]
        (is (= (string/trimr header) (:header-string result)))))))

(deftest read-header-string-rejects-truncated-length-field
  (doseq [[major nbytes expected]
          [[1 8 2]
           [1 9 2]
           [2 10 4]
           [2 11 4]
           [3 10 4]
           [3 11 4]]]
    (testing (str "format version " major " with " nbytes " bytes")
      (let [complete (npy-bytes major 0 "" StandardCharsets/ISO_8859_1)
            error    (header-error (Arrays/copyOf ^bytes complete (int nbytes)))]
        (is (= "Truncated .npy header length" (ex-message error)))
        (is (= {:expected expected :available (- nbytes 8)}
               (select-keys (ex-data error) [:expected :available])))))))

(deftest read-header-string-rejects-truncated-declared-header
  (doseq [[major charset] [[1 StandardCharsets/ISO_8859_1]
                           [2 StandardCharsets/ISO_8859_1]
                           [3 StandardCharsets/UTF_8]]]
    (testing (str "format version " major)
      (let [error (header-error (npy-bytes major 0 "abc" charset 10))]
        (is (= "Truncated .npy header" (ex-message error)))
        (is (= {:expected 10 :available 3}
               (select-keys (ex-data error) [:expected :available])))))))

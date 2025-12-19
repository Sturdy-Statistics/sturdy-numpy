(ns sturdy.numpy.test-utils
  (:require
   [clojure.java.io :as io]))

(def ^:private fixtures-dir "npy-fixtures")

(defn resource-path ^String [filename]
  (let [url (io/resource (str fixtures-dir "/" filename))]
    (when-not url
      (throw (ex-info "Missing test resource"
                      {:filename filename
                       :looked-for (str fixtures-dir "/" filename)})))
    (.getPath (io/file url))))

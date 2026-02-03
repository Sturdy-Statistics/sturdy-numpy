(ns hooks.sturdy.numpy.dataset
  (:require [clj-kondo.hooks-api :as api]))

(defn- make-defn
  "Build a (defn <name> [src rows cols] (do src rows cols nil)) node
   so clj-kondo won’t report unused bindings."
  [call name-sym]
  (let [m (meta call)
        tok (fn [s] (with-meta (api/token-node s) m))
        v   (fn [xs] (with-meta (api/vector-node xs) m))
        l   (fn [xs] (with-meta (api/list-node xs) m))]
    (l [(tok 'defn)
        (tok name-sym)
        (v [(tok 'src) (tok 'rows) (tok 'cols)])
        (l [(tok 'do) (tok 'src) (tok 'rows) (tok 'cols) (tok nil)])])))

(defn def-transposer
  "Hook for (def-transposer <fname> <src-tag> <array-ctor> <aset-fn>)
   We only care about <fname>."
  [{:keys [node]}]
  (let [[_ fname & _] (api/sexpr node)]
    (when-not (symbol? fname)
      (throw (ex-info "def-transposer: expected symbol as first arg" {:got fname})))
    {:node (make-defn node fname)}))

(defn def-splitter
  "Hook for (def-splitter <fname> <src-tag>)
   We only care about <fname>."
  [{:keys [node]}]
  (let [[_ fname & _] (api/sexpr node)]
    (when-not (symbol? fname)
      (throw (ex-info "def-splitter: expected symbol as first arg" {:got fname})))
    {:node (make-defn node fname)}))

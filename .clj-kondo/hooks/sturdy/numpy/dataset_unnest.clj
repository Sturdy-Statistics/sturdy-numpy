(ns hooks.sturdy.numpy.dataset-unnest
  (:require [clj-kondo.hooks-api :as api]))

(defn- make-defn-4
  "Build a (defn <name> [src rows cols fortran?] (do src rows cols fortran? nil))
   so clj-kondo registers the var and doesn’t warn about unused bindings."
  [call name-sym]
  (let [m (meta call)
        tok (fn [s] (with-meta (api/token-node s) m))
        v   (fn [xs] (with-meta (api/vector-node xs) m))
        l   (fn [xs] (with-meta (api/list-node xs) m))]
    (l [(tok 'defn)
        (tok name-sym)
        (v [(tok 'src) (tok 'rows) (tok 'cols) (tok 'fortran?)])
        (l [(tok 'do) (tok 'src) (tok 'rows) (tok 'cols) (tok 'fortran?) (tok nil)])])))

(defn def-unnester-skip-zeros
  "Hook for (def-unnester-skip-zeros <fname> <src-tag> <val-array-ctor> <aset-val> <zero-lit>)
   We only care about <fname> and the generated function arity."
  [{:keys [node]}]
  (let [[_ fname & _] (api/sexpr node)]
    (when-not (symbol? fname)
      (throw (ex-info "def-unnester-skip-zeros: expected symbol as first arg"
                      {:got fname})))
    {:node (make-defn-4 node fname)}))

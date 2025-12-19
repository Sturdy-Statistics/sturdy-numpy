(ns sturdy.numpy.dataset-list
  (:require
    [sturdy.numpy.read :refer [read-npy-primitive]]
    [tech.v3.dataset :as ds]
    [tech.v3.datatype :as dtype]))

(set! *warn-on-reflection* true)

(defn- unsigned-target [dtype]
  (case dtype
    :u1 :uint8
    :u2 :uint16
    :u4 :uint32
    nil))

(defn npy->dataset-rowlists
  "Read a 2D NumPy `.npy` file into a dataset with a single column `:c1`.

   Each row in `:c1` is a zero-copy view (dtype-next sub-buffer) of length
   `ncols`, representing that row in row-major order.

   Notes:
   - Only 2D arrays are supported.
   - C-order (row-major) files are supported.
   - Fortran-order files are not currently supported by this helper.
   - The resulting column has elemwise dtype `:object` (each cell is a buffer).
     Some downstream systems may not accept object columns for bulk ingestion."
  [path]
  (let [{:keys [shape dtype fortran? data]} (read-npy-primitive path)
        _ (when-not (= 2 (count shape))
            (throw (ex-info "npy->dataset-rowlists requires a 2D array"
                            {:shape shape})))
        _ (when fortran?
            (throw (ex-info "npy->dataset-rowlists does not currently support Fortran-order files"
                            {:shape shape})))
        [rows cols] shape

        ;; Wrap unsigned arrays so consumers can observe uint* element types.
        target (unsigned-target dtype)
        buf    (if target
                 (dtype/->array-buffer target data)
                 (dtype/->buffer data))

        rowviews (mapv (fn [r]
                         ;; zero-copy view into buf
                         (dtype/sub-buffer buf (* r cols) cols))
                       (range rows))]

    (ds/->dataset {:c1 rowviews})))

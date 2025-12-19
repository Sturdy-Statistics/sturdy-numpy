(ns sturdy.numpy
  (:require
   [sturdy.numpy.read :as r]
   [sturdy.numpy.dataset :as d]
   [sturdy.numpy.dataset-list :as l]
   [sturdy.numpy.dataset-unnest :as u]))

(defn npy->vec
  "Read a NumPy `.npy` file and return its contents as idiomatic Clojure data.

   This is a convenience wrapper around `sturdy.numpy.read/read-npy` that
   materializes the array as persistent Clojure collections.

   Returns a persistent vector (1D) or vector of vectors (2D), in row-major
   (C-order) layout.

   Notes:
   - The returned data is always in row-major order, regardless of whether
     the file is stored as C-order or Fortran-order.
   - This representation is easy to inspect and test, but incurs allocation
     and boxing costs for large arrays. Prefer `npy->dataset` for performance-
     sensitive or columnar workloads."
  [path]
  (r/read-npy path))

(defn npy->dataset
  "Read a NumPy `.npy` file and return its contents as a `tech.v3.dataset`.

   Behavior:
   - 1D arrays are returned as a single column dataset (`:c1`).
   - 2D arrays are returned as one dataset column per NumPy column.
   - Row-major (C-order) data is transposed during ingestion.
   - Column-major (Fortran-order) data is split without transposition.

   Column names are generated as `:c1`, `:c2`, … in column order.

   Dtypes:
   - Signed integer and float dtypes are preserved.
   - Unsigned NumPy dtypes (`:u1`, `:u2`, `:u4`) are represented with unsigned
     tech.datatype column dtypes (`:uint8`, `:uint16`, `:uint32`).

   This function avoids intermediate persistent collections and is the
   recommended entry point for large arrays or downstream analytics."
  [path]
  (d/npy->dataset path))

(defn npy->primitive
  "Read a NumPy `.npy` file and return its contents backed by primitive arrays.

   This represents the lowest-level access to `.npy` data provided by this
   namespace.

   Returns a map with:
   - `:shape`     Vector of dimensions (1D or 2D).
   - `:dtype`     Keyword identifying the NumPy dtype (e.g. `:f4`, `:i4`, `:u2`).
   - `:fortran?`  Boolean indicating whether the on-disk layout is column-major
                  (Fortran-order) or row-major (C-order).
   - `:data`      Flat Java primitive array containing the decoded values,
                  laid out according to `:fortran?`.

   No transposition, reshaping, or conversion to persistent collections is
   performed. Callers are responsible for interpreting the layout correctly.

   This function is intended for high-performance ingestion paths and serves
   as the foundation for `npy->dataset`."
  [path]
  (r/read-npy-primitive path))

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
  (l/npy->dataset-rowlists path))

(defn npy->dataset-unnested-nz
  "Read a NumPy `.npy` file and return a sparse, long-form `tech.v3.dataset`.

   This function produces an \"UNNEST\"-style representation with one row per
   non-zero entry in the array, using three columns:

   - `:row_no` — 0-based row index (`int64`)
   - `:col_no` — 0-based column index (`int16`)
   - `:val`    — value at `(row_no, col_no)`, primitive-backed with dtype preserved

   Behavior:
   - Only non-zero values are emitted (zero entries are skipped).
   - 1D arrays are treated as shape `(nrows, 1)`.
   - 2D arrays are supported.
   - Both C-order (row-major) and Fortran-order (column-major) files are handled
     correctly without transposition.
   - The physical order of rows in the result is unspecified; `(row_no, col_no)`
     uniquely identifies each entry.

   Dtypes:
   - Signed integer and floating-point dtypes are preserved.
   - Unsigned NumPy dtypes (`:u1`, `:u2`, `:u4`) are represented using unsigned
     `tech.datatype` column dtypes (`:uint8`, `:uint16`, `:uint32`).

   Performance characteristics:
   - Two-pass ingestion (count non-zeros, then allocate and populate arrays).
   - Avoids materializing dense columnar representations.
   - No boxing in hot loops.
   - Memory usage proportional to the number of non-zero entries.

   This representation is especially well-suited for:
   - Extremely sparse arrays
   - Fast ingestion into databases such as DuckDB
   - Downstream aggregation into sparse row-wise or list-based formats

   If the array is dense or column-oriented access is required, prefer
   `npy->dataset`."
  [path]
  (u/npy->dataset-unnested-nz path))

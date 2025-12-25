# sturdy.numpy

A small, focused Clojure library for reading NumPy `.npy` files.

[![Clojars Project](https://img.shields.io/clojars/v/com.sturdystats/sturdy-numpy.svg)](https://clojars.org/com.sturdystats/sturdy-numpy)

```clj
com.sturdystats/sturdy-numpy {:mvn/version "VERSION"}
```

## Features

- Reads NumPy `.npy` files (binary array format)
- Supports **1D and 2D arrays**
- Supports numeric dtypes:
  - Unsigned integers: `u1`, `u2`, `u4`
  - Signed integers: `i1`, `i2`, `i4`, `i8`
  - Floating point: `f4`, `f8`
- Supports **little-endian and big-endian** files
- Supports **C-order and Fortran-order** layouts
- Multiple output representations:
  - Idiomatic Clojure data (`vector` / `vector-of-vectors`)
  - `tech.v3.dataset` (columnar, primitive-backed)
  - Raw Java primitive arrays (lowest-level access)

## Example Usage

### Clojure Vectors

```clj
(require '[sturdy.numpy :as np])

(let [f "test-resources/npy-fixtures/shape_2x3__dtype_u4.npy"]
  (np/npy->vec f))

;; => [[4294967291 12 29] [46 63 80]]
```

These are
- Always returned in row-major (C-order) layout
- Easy to inspect and test
- Not optimized for large arrays

### `tech.v3.dataset`

```clj
(require '[sturdy.numpy :as np])

(let [f "test-resources/npy-fixtures/shape_2x3__dtype_u4.npy"]
  (np/npy->dataset f))

;; => _unnamed [2 3]:
;;    |        :c1 | :c2 | :c3 |
;;    |-----------:|----:|----:|
;;    | 4294967291 |  12 |  29 |
;;    |         46 |  63 |  80 |
```

- 1D arrays → single-column dataset (`:c1`)
- 2D arrays → one column per NumPy column
- Column names are generated as `:c1`, `:c2`, … reflecting the order in the original file

This is the recommended entry point for:
- large arrays
- database ingestion (e.g. DuckDB)
- most tasks

### Preserves `dtype`

```clj
(require '[sturdy.numpy :as np])
(require '[tech.v3.dataset :as ds])
(require '[tech.v3.datatype :as dtype])

(let [f       "test-resources/npy-fixtures/shape_2x3__dtype_u4.npy"
      dataset (np/npy->dataset f)]
  (map dtype/elemwise-datatype (ds/columns dataset)))

;; => (:uint32 :uint32 :uint32)
```

```clj
(require '[sturdy.numpy :as np])
(require '[tech.v3.dataset :as ds])
(require '[tech.v3.datatype :as dtype])

(defn get-dtype [dataset]
  (-> (ds/columns dataset) first dtype/elemwise-datatype))

(defn test-file [dtype fname]
  (let [dataset (np/npy->dataset fname)]
    {:expected dtype
     :actual (get-dtype dataset)}))

(for [tp [:u1 :u2 :u4 :i1 :i2 :i4 :i8 :f4 :f8]]
  (let [fname (format "test-resources/npy-fixtures/shape_2x3__dtype_%s.npy"
                      (name tp))]
    (test-file tp fname)))
;; => ({:expected :u1, :actual :uint8}
;;     {:expected :u2, :actual :uint16}
;;     {:expected :u4, :actual :uint32}
;;     {:expected :i1, :actual :int8}
;;     {:expected :i2, :actual :int16}
;;     {:expected :i4, :actual :int32}
;;     {:expected :i8, :actual :int64}
;;     {:expected :f4, :actual :float32}
;;     {:expected :f8, :actual :float64})
```

### Primitive Arrays (advanced)

```clj
(require '[sturdy.numpy :as np])

(np/npy->primitive "shape_2x3__dtype_u4.npy")

;; => {:shape    [2 3]
;;     :dtype    :u4
;;     :fortran? false
;;     :data     #<long[]>}
```
- Returns a flat Java primitive array
- Layout depends on :fortran?
- No reshaping or transposition is performed

This API is intended for:
- custom ingestion pipelines
- zero-copy workflows
- advanced performance-sensitive use cases

### Experimental: Row-list Datasets

For some workflows (e.g. downstream databases that support list or array types), it can be useful to represent each row of a 2D NumPy array as a single list-valued column rather than as many scalar columns.

The function `npy->dataset-rowlists` provides this representation:

```clj
(require '[sturdy.numpy :as np])

(np/npy->dataset-rowlists "test-resources/npy-fixtures/shape_2x3__dtype_i4.npy")
;; => _unnamed [2 1]:
;;    |        :c1 |
;;    |------------|
;;    | [-5 12 29] |
;;    | [46 63 80] |

(np/npy->dataset-rowlists "test-resources/npy-fixtures/shape_2x3__dtype_f4.npy")
;; => _unnamed [2 1]:
;;    |               :c1 |
;;    |-------------------|
;;    | [-3.0 -1.75 -0.5] |
;;    |   [0.75 2.0 3.25] |
```

Each row is represented as a zero-copy buffer view over the underlying array data.

Notes:
- Only 2D arrays are supported.
- Row-major (C-order) `.npy` files are supported.
- Fortran-order files are not currently supported by this helper.
- The resulting column has element dtype `:object` (each cell is a buffer), which may not be accepted by all ingestion paths.
- This helper is experimental and primarily intended for advanced ingestion pipelines or custom database integrations.

### Experimental: Sparse `UNNEST`ed Datasets

For **very sparse arrays**, a column-oriented dataset with one column per NumPy column can be inefficient: most values are zero, and downstream systems often want a sparse or list-based representation anyway.

The function `npy->dataset-unnested-nz` provides an alternative representation inspired by SQL `UNNEST` / long-form tables.

Instead of producing one column per NumPy column, it produces a row-wise sparse representation with three columns:
- `:row_no` — row index (0-based, `int64`)
- `:col_no` — column index (0-based, `int16`)
- `:val` — value (primitive-backed, `dtype` preserved)
Only non-zero entries are emitted.

```clj
(npy->dataset-unnested-nz "test-resources/npy-fixtures/shape_2x3__dtype_u4.npy")
;; => _unnamed [6 3]:
;;    | :row_no | :col_no |       :val |
;;    |--------:|--------:|-----------:|
;;    |       0 |       0 | 4294967291 |
;;    |       0 |       1 |         12 |
;;    |       0 |       2 |         29 |
;;    |       1 |       0 |         46 |
;;    |       1 |       1 |         63 |
;;    |       1 |       2 |         80 |
```

For Fortran-order (`order='F'`) `.npy` files, the physical order of rows differs, but `(row_no, col_no)` are computed correctly:

```clj
(npy->dataset-unnested-nz "test-resources/npy-fixtures/shape_2x3__dtype_u4__order_F.npy")
;; => _unnamed [6 3]:
;;    | :row_no | :col_no |       :val |
;;    |--------:|--------:|-----------:|
;;    |       0 |       0 | 4294967291 |
;;    |       1 |       0 |         46 |
;;    |       0 |       1 |         12 |
;;    |       1 |       1 |         63 |
;;    |       0 |       2 |         29 |
;;    |       1 |       2 |         80 |

```

The row order is not significant; consumers should treat the dataset as an unordered collection of `(row, col, val)` triples.

#### Why this format?
This representation is especially useful when:
- The array is extremely sparse
- The column count is large
- You intend to ingest directly into a database such as DuckDB
- You want to immediately aggregate into list- or sparse-row formats

For example, in DuckDB you might do:
```sql
SELECT
  row_no,
  list(col_no ORDER BY col_no) AS inds,
  list(val    ORDER BY col_no) AS vals
FROM staging
GROUP BY row_no;
```

This yields a compact per-row sparse representation suitable for downstream modeling or analytics.

#### Performance characteristics
- Two-pass algorithm:
  1. Count non-zero entries
  2. Allocate exactly-sized primitive arrays and populate them
- No transposition or per-column materialization
- No boxing in hot loops
- Preserves original NumPy dtype (including unsigned integers)
- `col_no` is stored as `int16` (columns < 32k)
- `row_no` is stored as `int64` (supports millions of rows)

This makes `npy->dataset-unnested-nz` significantly more memory-efficient than dense columnar datasets when sparsity is high.

#### Notes and limitations
- Only **1D and 2D** arrays are supported
- Zero is defined as:
  - `0` for integer types
  - `0.0` / `-0.0` for floating-point types (exact comparison)
- NaNs are **not** treated as zero
- Row and column indices are **0-based**

## Non-goals (v0.1.0)

- Higher-dimensional arrays (3D+)
- Structured /record dtypes
- Memory-mapped or streaming IO
- Writing `.npy` files

## License

Apache License 2.0

Copyright © Sturdy Statistics

<!-- Local Variables: -->
<!-- fill-column: 10000000 -->
<!-- End: -->

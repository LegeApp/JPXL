# I.9.6 / I.9.7: where the two half-blocks land

Date: 2026-08-03. Slice 8A (VarDCT varblock vocabulary and DCT engine).
Not an oracle experiment: settled from the normative text by cross-clause
consistency, before any VarDCT pixel could be decoded.

## 1. Question

I.9.6 reconstructs a DCT8x4 varblock by "dividing [the 8x8 samples] into two
8x4 vertical blocks `samples_8x4`", and I.9.7 does the same for DCT4x8 with
"two 4x8 horizontal blocks `samples_4x8`". Both clauses build a temporary
4-row by 8-column coefficient matrix, run `IDCT_2D` on it, and store the result
in `samples_8x4[x]` / `samples_4x8[y]` — and then **stop**. Neither clause says
where those two half-blocks go in the 8x8 output.

The slice-8 scoping report also flagged the two clauses as possibly
inconsistent, since their gathers are textually identical apart from the loop
variable name.

## 2. Preregistered gate

* **Settled** — the normative text elsewhere fixes the placement unambiguously,
  and the resulting reading is consistent with both clauses' shapes and with
  the `dcs` butterfly.
* **Unsettled** — no such text; ship a flip point and defer to an oracle probe
  in wave 3.

Fixed before reading I.9.8.

## 3. Method

Read I.9.6, I.9.7 and I.9.8 in `latex/part1.tex` (canonical for Part 1) and
cross-checked every line against `markdowns/standard-markdowns/part1.md`. Then
compared the sub-block gathers of the three clauses term by term.

## 4. Raw results

**The gathers are not inconsistent; they are the same code, correctly.** Both
clauses split the 8x8 coefficient block by *row parity*: half `h` takes the
full 8 columns of rows `h`, `h + 2`, `h + 4`, `h + 6`, and replaces the DC with
entry `h` of the two-element sum/difference pair built from the coefficients at
`(0, 0)` and `(0, 1)`. So half 0 reads rows 0, 2, 4, 6 and half 1 reads rows
1, 3, 5, 7. Each half's
coefficient matrix is 4 rows by 8 columns, which is the landscape shape I.3.2
requires for both an 8x4 and a 4x8 sample block. The clauses differ only in the
`(R, C)` they hand `IDCT_2D`: `(8, 4)` for DCT8x4, `(4, 8)` for DCT4x8. The
`coeffs_8x4(0, 0) = dcs[x]` line in I.9.6 is a naming slip present in all three
transcriptions; the temporary is declared as `coeffs_4x8` and used as
`coeffs_4x8` two lines later.

**I.9.8 supplies the missing placement.** Its 4x8 sub-block is gathered from
rows 1, 3, 5, 7 with its DC set to the *difference* of the coefficients at
`(0, 0)` and `(0, 1)` — which is *exactly* I.9.7's half 1: the same odd rows,
the same difference DC. And I.9.8 does state where that sub-block goes: at rows
4..8 when `flip_y` is 0, and rows 0..4 when it is 1. So in the unflipped case
half index 1 sits at the high coordinates, and half index 0 at the low ones.

**A second, independent consistency argument.** I.9.3's `AuxIDCT2x2` writes the
all-plus butterfly output `r00 = c00 + c01 + c10 + c11` to the *low* position
`(x*2, y*2)`. `dcs[0] = c(0,0) + c(0,1)` is the same construction in one
dimension, and it likewise belongs at the low coordinate.

**The DCT8x4 side** has no AFV analogue, but it is fixed by the column/row
symmetry of the clause pair: the halves are indexed by `x` in I.9.6 and by `y`
in I.9.7, the outputs are 8x4 ("vertical", stacked side by side) and 4x8
("horizontal", stacked one above the other), and the two reconstructions must
be each other's transpose because their inputs and gathers are.

## 5. Conclusion

Settled from the normative text: half index `h` occupies columns `4h..4h+4` for
DCT8x4 and rows `4h..4h+4` for DCT4x8.

What this does **not** establish: it is an argument from the internal
consistency of I.9.3, I.9.7 and I.9.8, not a statement I.9.6 or I.9.7 makes
directly. The DCT8x4 half of the argument rests on the transpose symmetry of
the clause pair rather than on a placement I.9.8 states. A wave-3 oracle probe
against `djxl` on a fixture containing DCT8x4 varblocks is still worth running,
and would be decisive.

## 6. Consequences

* `jpxl-core::varblock::DCT8X4_HALF_INDEX_IS_LOW_COORDINATE` (default `true`)
  carries both readings; flipping it swaps the two halves for both clauses.
* `dct8x4_half_placement_snapshot` pins the placement (it fails when the
  constant is flipped); `dct8x4_is_the_transpose_of_dct4x8` pins the symmetry
  argument (it passes under either setting, by design — the two clauses stay
  transposes of each other however the halves are ordered, which is why the
  snapshot test is the one that matters).
* Slice 8F inherits the probe, not a blocker.

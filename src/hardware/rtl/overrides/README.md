# CV-Wally overrides

To change an upstream CV-Wally file, copy it here unmodified, commit, then edit:

```bash
cp src/hardware/cvw/src/ieu/alu.sv src/hardware/rtl/overrides/
```

The hardware `Makefile` drops any `cvw/src/**/<name>.sv` that has a same-named file in this
directory and compiles this copy instead. Lint and simulation both pick it up.
Committing the unmodified copy first means `git diff` shows exactly what the project changed.

Upstream file names must stay unique for this to work (they currently are).

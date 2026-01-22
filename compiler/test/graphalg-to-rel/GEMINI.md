# GraphAlg to Relation Algebra tests
These test files verify the `graphalg-to-rel` pass, which converts ops from the `graphalg` dialect into `garel` and `arith` dialect ops.

## Running Tests
Tests require the `graphalg-opt` binary, which is built by running `cmake --build compiler/build --target graphalg-opt`.
To get the output for a test file, run `./compiler/build/tools/graphalg-opt --graphalg-to-rel compiler/test/graphalg-to-rel/<name>.mlir`.
Test files contain `CHECK` comments that are verified using LLVM's FileCheck tool, installed as `FileCheck-20`.

If you make any changes and have verified that the individual tests are correct, run the integration tests as a final check: `cmake --build compiler/build --target check`.

## Coding style
### Use CHECK-LABEL For independent test functions
```mlir
// CHECK-LABEL: @AddBool
func.func @AddBool(%arg0: !graphalg.mat<1 x 1 x i1>, %arg1: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x i1> {
    ...
}

// CHECK-LABEL: @AddInt
func.func @AddInt(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
    ...
}
```

### Keep new op CHECKs close to original ops
Keep CHECK comments for output ops directly before and at the same indentation as the original ops they were generated from.

**GOOD**:
```mlir
// CHECK-LABEL: @AddBool
func.func @AddBool(%arg0: !graphalg.mat<1 x 1 x i1>, %arg1: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x i1> {
  // CHECK: %[[#PROJECT:]] = garel.project {{.*}} : <i1, i1> -> <i1>
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x i1>, !graphalg.mat<1 x 1 x i1> -> <1 x 1 x i1> {
  ^bb0(%arg2 : i1, %arg3: i1):
    // CHECK: %[[#LHS:]] = garel.extract 0
    // CHECK: %[[#RHS:]] = garel.extract 1
    // CHECK: %[[#ADD:]] = arith.ori %[[#LHS]], %[[#RHS]]
    %1 = graphalg.add %arg2, %arg3 : i1

    // CHECK: garel.project.return %[[#ADD]]
    graphalg.apply.return %1 : i1
  }

  // CHECK: return %[[#PROJECT]]
  return %0 : !graphalg.mat<1 x 1 x i1>
}
```

**BAD**:
```mlir
// CHECK-LABEL: @AddBool
// CHECK: %[[#PROJECT:]] = garel.project {{.*}} : <i1, i1> -> <i1>
// CHECK: %[[#LHS:]] = garel.extract 0
// CHECK: %[[#RHS:]] = garel.extract 1
// CHECK: %[[#ADD:]] = arith.ori %[[#LHS]], %[[#RHS]]
// CHECK: garel.project.return %[[#ADD]]
// CHECK: return %[[#PROJECT]]
func.func @AddBool(%arg0: !graphalg.mat<1 x 1 x i1>, %arg1: !graphalg.mat<1 x 1 x i1>) -> !graphalg.mat<1 x 1 x i1> {
  %0 = graphalg.apply %arg0, %arg1 : !graphalg.mat<1 x 1 x i1>, !graphalg.mat<1 x 1 x i1> -> <1 x 1 x i1> {
  ^bb0(%arg2 : i1, %arg3: i1):
    %1 = graphalg.add %arg2, %arg3 : i1
    graphalg.apply.return %1 : i1
  }

  return %0 : !graphalg.mat<1 x 1 x i1>
}
```

## Porting IPR tests
If you are asked to port an IPR testcase, do these things:
1. Change ag-opt in the `RUN` comment to graphalg-opt and the pass from --graphalg-to-ipr to --graphalg-to-rel
2. Run `./compiler/build/tools/graphalg-opt --graphalg-to-rel compiler/test/graphalg-to-rel/<name>.mlir` to see the expected output. Use that to guide the changes described in (3) and (4).
3. Replace `ipr.tuplestream` types with the corresponding `garel.relation`, and `ipr.tuple` with `garel.tuple`
4. Replace `ipr.*` ops with `garel.*` or `arith.*` ops.
5. Verify your changes with `FileCheck-20` (see guide to running tests above).
6. When you have verified your changes to the file, run the integration tests to double-check.

Do not make changes to the input IR (the parts not in comments).
If you really think this is necessary, ask first.

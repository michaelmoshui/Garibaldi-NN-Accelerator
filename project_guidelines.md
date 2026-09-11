# Term Project: Quantized Neural-Network Acceleration on RV64IMC

## Project Overview

In this project, a team of will design and evaluate architectural extensions for accelerating quantized neural-network inference on a **bare-metal RV64IMC RISC-V processor**, using CV-Wally RISC-V core.

The project combines workload characterization, ISA design, software optimization, architectural simulation, RTL implementation, and quantitative evaluation.

The baseline processor is:

- RV64IMC
- Bare-metal execution
- No floating-point or vector extension
- INT8 weights
- INT8 or INT16 activations

The primary objective is to determine which small architectural extensions provide the best performance improvement for quantized edge-AI workloads while maintaining reasonable hardware cost.

---

## 1. Model Candidates

Each team should select one primary model of approximately **2–12 million parameters** and approximately **0.2–2 GMAC per inference**.

Recommended candidates include:

| Model | Approx. Parameters | Approx. Compute | Characteristics |
|---|---:|---:|---|
| **DeiT-Tiny** | 5.7 M | ~1.3 GMAC | Transformer; GEMM-heavy |
| **MobileViT** | 4.9 M | ~1.8 GMAC | CNN/Transformer hybrid |
| **EfficientFormer-L1** | ~6 M | ~1.3 GMAC | Efficient mobile Transformer |
| **LeViT-128S** | ~8 M | ~0.8 GMAC | Efficient attention architecture |
| **MobileNetV3-Large** | 5.4 M | ~0.22 GMAC | Convolution-heavy baseline |
| **EfficientNet-Lite0** | 4.7 M | ~0.39 GMAC | Mobile CNN |
| **ResNet-18** | 11.7 M | ~1.8 GMAC | Traditional convolutional network |

Transformer or hybrid models are preferred because their dominant operations map naturally to packed integer dot-product instructions. You may download models from https://huggingface.co/litert-community. Or you may convert other downloaded models from Pytorch to LiteRT. It's recommended to quantize and port baremetal LiteRT models.

The selected model must be quantized to at least:

- **W8A8: INT8 weights + INT8 activations**

Optional:

- **W8A16: INT8 weights + INT16 activations**

The implementation does not need to support a general-purpose ML runtime. Model parameters and graph information may be converted offline into static C data structures for bare-metal execution.

---

## 2. Required Project Components

### A. Bare-Metal Baseline

Implement the selected quantized network, or all major computational kernels required by it, on RV64IMC.

The baseline must include:

- Quantized matrix multiplication / fully connected kernels
- Convolution kernels if required by the model
- Requantization
- Required nonlinear or normalization operations
- Static memory allocation
- No floating-point hardware dependence

At minimum, the team must demonstrate correct execution of the dominant computational kernels.

Full-model inference is strongly preferred.

### B. Workload and Architecture Analysis

Use **Spike or another RISC-V ISA simulator** to characterize the workload before introducing architectural changes.

Required measurements include:

1. Total dynamic instruction count
2. Instruction mix:
   - load
   - store
   - integer ALU
   - multiply
   - branch
3. Dynamic memory instructions
4. MAC operations per retired instruction
5. Execution contribution of major operators
6. Code size
7. Custom-instruction utilization after ISA modification

The team must identify at least the **three largest architectural bottlenecks**.

Examples include:

- Excessive scalar multiply instructions
- Load/store instruction overhead
- Requantization overhead
- Cache misses
- Branch/loop overhead
- Load-use stalls
- Insufficient memory bandwidth

Architectural extensions should be justified by measurements rather than selected in advance without profiling.

---

## 3. Required ISA Extension

The minimum required architecture extension are a packed integer MAC and requantization instructions.

Example:

### `Xqmac8`

One instruction performs approximately:

**8 × INT8 multiply-accumulate operations**

using packed 64-bit operands.

Conceptually:

```text
acc +=
 A0 × W0 +
 A1 × W1 +
 ...
 A7 × W7
```

### `Requantization Instruction`

Accelerate:

```text
accumulator of fixed-point multiply
→ rounding
→ shift
→ zero-point addition
→ saturation
```

Evaluate:

- Instructions/output element
- Cycles/output element
- End-to-end speedup

Evaluation of instruction effectivness should be based incremental implementation of one instruction on the other.

The team must:

- Define instruction semantics
- Implement the instruction in Spike or an equivalent simulator
- Modify one or more inference kernels to use it
- Verify results against a software reference
- Measure instruction-count reduction

The exact instruction encoding and accumulator width are design decisions that must be documented.

---

## 4. Architecture Performance Evaluation

Spike should be used primarily for **ISA-level analysis**, not as a cycle-accurate representation of CV-Wally.

The project must distinguish:

- **ISA improvement**
- **Microarchitectural performance improvement**

Required comparisons:

| Configuration | Description |
|---|---|
| B0 | Baseline RV64IMC |
| B1 | Software-optimized RV64IMC |
| X1 | RV64IMC + Xqmac |
| X2 | Additional extension, if implemented |

Required ISA-level metrics:

- Dynamic instruction reduction
- MAC/instruction
- Load/store instructions per MAC
- Custom-instruction frequency
- Code-size change

At least one representative kernel must also be evaluated using CV-Wally RTL.

Required hardware-level metrics:

- Cycle count
- CPI
- Speedup
- Maximum clock frequency or critical path
- Area/resource overhead. The area estimation should be based on Synopsys DC or RTL architect with SAED 32/28nm cell libraries.

Recommended final metric:

**Performance improvement versus hardware cost: speedup/area**


## 5. Required Quantitative Targets

A successful project should demonstrate all of the following.

### Functional

- Quantized kernel output matches the reference implementation.
- Xqmac passes directed and randomized verification tests.
- No unintended floating-point instruction is required.

### Performance

The team must demonstrate at least one of:

- **≥2× reduction in dynamic instructions** for a major GEMM/FC kernel, or
- **≥2× kernel-level cycle speedup**, or
- **≥1.5× end-to-end model speedup**

The project is not required to meet all three targets.

If performance improvement is lower, the team must provide quantitative analysis explaining the limiting bottleneck.

### Evaluation Coverage

At minimum report results for:

- One microbenchmark
- Two major model operators
- Complete model inference or one complete Transformer/CNN block

---

## 6. Work list



- Model workload analysis
- Spike integration
- Xqmac ISA design
- Instruction profiling
- Trace analysis
- Performance modeling
- ISA design-space exploration
- Bare-metal execution
- Optimized kernels
- CV-Wally RTL implementation
- Verification
- Hardware counters
- Synthesis / FPGA evaluation

---

## 7. Architecture Extensions

Part of credits are earned by implementing and quantitatively evaluating additional architectural features.

Extensions should preferably be motivated by bottlenecks observed after introducing Xqmac and requantization.

Students can propose and implement other architecture extensions.

### Compute Extensions

#### W8A16 Support

Support:

**INT16 activations × INT8 weights**

and compare accuracy and performance with W8A8.

### Memory-System Extensions

#### Post-Increment Load/Store  (**★★** +10%)

Combine:

```text
load
pointer increment
```

into one instruction.

Evaluate reduction in address-generation instructions.

#### Pipelined Cache (**★★★** +15%)

Allow a new cache access every cycle even when cache latency exceeds one cycle.

Measure:

- Load throughput
- Stall cycles
- MAC utilization

#### Multi-Banked Cache (**★★★** +15%)

Implement two or more independent cache banks.

Measure:

- Bank conflicts
- Effective memory accesses/cycle
- Inference speedup

#### Nonblocking Cache (**★★★★** +20%) 

Support hit-under-miss or multiple outstanding misses.

Measure:

- Memory-stall cycles
- Outstanding misses
- Total execution time

#### Hardware Prefetching / Stream Buffer (**★★★** +15%) 

Examples:

- Next-line prefetch
- Stride prefetch
- Weight stream buffer

Measure:

- Prefetch accuracy
- Coverage
- Additional memory traffic
- Cycle reduction

#### Software-Managed Scratchpad (**★★★** +15%) 

Compare a small scratchpad against a similarly sized cache.

Measure:

- Data reuse
- Memory traffic
- Performance
- Hardware cost

### Pipeline / Issue Extensions

#### Limited Dual Issue (**★★★★** +20%)

Allow one instruction from each class to execute concurrently:

```text
Slot 0: load/store or integer operation
Slot 1: Xqmac
```

A full general-purpose superscalar processor is not required.

Evaluate:

- Issued instructions/cycle
- Xqmac utilization
- Pipeline stalls
- Speedup
- Area overhead

#### Fused Load + MAC (**★★★** +15%) 

Explore an instruction that combines memory access with packed MAC execution.

Because such instructions complicate memory exceptions and pipeline control, teams implementing this feature must clearly specify architectural semantics.

#### Hardware Loops (**★★★** +15%) 

Eliminate branch and loop-counter instructions in regular GEMM loops.

Measure dynamic instruction reduction and branch-related cycles.

### Cache Design Exploration (**★** +5%)

Teams may compare:

- Cache sizes
- Associativities
- Cache-line sizes
- Random / LRU / pseudo-LRU replacement

A cache-policy-only project extension should demonstrate measurable workload sensitivity rather than simply implementing multiple policies.

---

## 8. Final Deliverables

Each team must submit:

1. Source code for the bare-metal workload.
2. Quantized model or model-conversion scripts.
3. Spike/custom ISA implementation.
4. CV-Wally RTL changes or equivalent hardware implementation.
5. Verification tests.
6. Reproducible benchmark scripts.
7. Performance measurements.
8. Architecture design-space analysis.
9. Final presentation.
10. Approximately **6–8 page conference-style report**.

The final report should clearly answer:

- **What is the workload bottleneck?**
- **Which architectural feature addresses it?**
- **How much performance does the feature provide?**
- **What hardware cost is required?**
- **What becomes the next bottleneck after the optimization?**

The strongest projects will present a progression such as:

```text
RV64IMC baseline
        ↓
software optimization
        ↓
Xqmac / requantization
        ↓
new bottleneck identified
        ↓
memory / pipeline extension
        ↓
performance-area evaluation
```

The goal is not simply to build the fastest accelerator, but to demonstrate a rigorous:

**measurement → architecture proposal → implementation → evaluation** process.

The suggested GitHub repo structure:

README.md
docs/
experiments/
results/
scripts/
src/

DESIGN.md
EXPERIMENTS.md
AI\_USAGE.md

## 9. Grading Policy

We will have two-stage evaluations.

1. 20pt Midterm report (18pt) and presentation (2pt): ISA and Spike with custom instructions.
2. 20pt*80% Final report (14.4pt) and presentation (1.6pt): CV Wally RTL implementation.
3. 20pt*20% Extension (4pt)

Reference evaluation forms for the report:

| Items                        | Weight | Focus                                                       |
| ---------------------------- | -----: | ----------------------------------------------------------- |
| Functional correctness       |    20% | tests, reference matching, ISA semantics                    |
| Architecture analysis        |    20% | Spike profiling, bottleneck identification, instruction mix |
| ISA/microarchitecture design |    20% | rationale, implementation quality, tradeoffs                |
| Quantitative evaluation      |    20% | speedup, CPI, area, Fmax, reproducibility                   |
| Report quality               |    15% | clarity, figures, evidence-backed conclusions               |
| Reproducibility              |     5% | scripts, configuration, rerunnable experiments              |


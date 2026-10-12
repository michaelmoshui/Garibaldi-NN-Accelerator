# Bare-metal startup for the CV-Wally testbench (garibaldi_rv64imc).
#
# Reset enters _start at 0x80000000 in M-mode with the program image already
# preloaded into IROM and DTIM by the testbench. _start sets sp/gp and mtvec,
# clears .bss, runs static constructors, calls main(), then ends the
# simulation with a 32-bit store to tohost (the testbench matches the "SW"
# mnemonic, so it must not be compressed or widened to sd).

  .section .text.init, "ax", @progbits
  .option norvc
  .globl _start
_start:
  .option push
  .option norelax
  la    gp, __global_pointer$
  .option pop
  la    sp, _stack_top
  la    t0, trap_handler
  csrw  mtvec, t0

  la    t0, __bss_start
  la    t1, __bss_end
1:
  bgeu  t0, t1, 2f
  sd    zero, 0(t0)
  addi  t0, t0, 8
  j     1b
2:

  la    s0, __init_array_start
  la    s1, __init_array_end
3:
  bgeu  s0, s1, 4f
  ld    t0, 0(s0)
  jalr  t0
  addi  s0, s0, 8
  j     3b
4:

  call  main
  mv    s0, a0

finish:
  la    t0, tohost
  sw    s0, 0(t0)
5:
  j     5b

# Any trap (illegal instruction, misaligned access, ecall, ...) is fatal:
# record it in selfcheck_record as {0x100 + mcause, mepc, mtval, ra, sp} and
# exit.
  .align 2
trap_handler:
  la    t0, selfcheck_record
  sd    ra, 24(t0)
  sd    sp, 32(t0)
  csrr  t1, mepc
  sd    t1, 8(t0)
  csrr  t1, mtval
  sd    t1, 16(t0)
  csrr  t1, mcause
  addi  t1, t1, 0x100
  sd    t1, 0(t0)
  li    s0, 1
  j     finish

# Replace the libstdc++/newlib versions, which drag in stdio, sprintf and the
# exit path (__libc_fini_array needs crti's _fini, absent with -nostartfiles).
# A pure virtual call traps (mcause 3); static destructors never run.
  .section .text.__cxa_pure_virtual, "ax", @progbits
  .globl __cxa_pure_virtual
__cxa_pure_virtual:
  ebreak

  .section .text.atexit, "ax", @progbits
  .globl atexit
atexit:
  li    a0, 0
  ret

  .section .tohost, "aw", @progbits
  .align 6
  .globl tohost
tohost:
  .dword 0
  .align 6
  .globl fromhost
fromhost:
  .dword 0

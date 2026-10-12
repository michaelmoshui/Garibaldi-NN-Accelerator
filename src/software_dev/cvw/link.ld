/* garibaldi_rv64imc memory map: IROM and DTIM both span 0x80000000-0x807FFFFF
 * and both are preloaded with the whole image, so code, read-only data and
 * read-write data share one region. The stack grows down from the top.
 * dtim.sv sizes its array as DTIM_RANGE/8 = 0xFFFFF doublewords, so the last
 * doubleword (0x807FFFF8) does not exist: stores are dropped, loads return 0.
 * LENGTH stops 16 bytes short to keep it (and the stack top) out of use. */
OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 8M - 16
}

SECTIONS
{
  .text : {
    KEEP(*(.text.init))
    *(.text .text.*)
  } > RAM

  .rodata : ALIGN(16) {
    *(.rodata .rodata.*)
    *(.srodata .srodata.*)
  } > RAM

  .init_array : ALIGN(8) {
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
    KEEP(*(.init_array .ctors))
    PROVIDE_HIDDEN(__init_array_end = .);
  } > RAM

  .tohost : ALIGN(64) {
    KEEP(*(.tohost))
  } > RAM

  .selfcheck : ALIGN(64) {
    KEEP(*(.selfcheck))
  } > RAM

  .data : ALIGN(16) {
    *(.data .data.*)
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > RAM

  .bss (NOLOAD) : ALIGN(16) {
    __bss_start = .;
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(8);
    __bss_end = .;
  } > RAM

  _end = .;
  PROVIDE(end = .);
  _stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
